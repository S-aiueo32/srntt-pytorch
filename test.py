import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from models import Swapper, VGG, SRNTT
from datasets import CUFED5Dataset
from losses import PSNR

TARGET_LAYERS = ['relu3_1', 'relu2_1', 'relu1_1']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='./data/CUFED5')
    parser.add_argument('--weight', '-w', type=str, required=True)
    parser.add_argument('--use_weights', action='store_true')
    args = parser.parse_args()

    dataset = CUFED5Dataset(args.dataroot)
    dataloader = DataLoader(dataset)

    vgg = VGG(model_type='vgg19').to(device)
    swapper = Swapper().to(device)
    model = SRNTT(use_weights=args.use_weights).to(device)
    model.load_state_dict(torch.load(args.weight))

    criterion_psnr = PSNR()

    table = []
    tbar = tqdm(total=len(dataloader))
    for batch_idx, batch in enumerate(dataloader):
        with torch.no_grad():
            img_hr = batch['img_hr'].to(device)
            img_lr = batch['img_lr'].to(device)
            img_in_up = batch['img_in_up'].to(device)

            map_in = vgg(img_in_up, TARGET_LAYERS)

            row = [batch['filename'][0].split('_')[0]]
            for ref_idx in range(7):
                ref = batch['ref'][ref_idx]
                map_ref = vgg(ref['ref'].to(device), TARGET_LAYERS)
                map_ref_blur = vgg(ref['ref_blur'].to(device), TARGET_LAYERS)

                maps, weights, correspondences = swapper(
                    map_in, map_ref, map_ref_blur)

                maps = {k: torch.tensor(v).unsqueeze(0).to(device)
                        for k, v in maps.items()}
                weights = torch.tensor(weights).to(device)
                weights = weights.reshape(1, 1, *weights.shape)

                _, img_sr = model(img_lr, maps, weights)

                name = f'{batch["filename"][0]}_{ref_idx}.png'
                save_image(img_sr.clamp(0, 1), Path(args.weight).parent / name)

                psnr = criterion_psnr(img_sr.clamp(0, 1), img_hr.clamp(0, 1))
                row.append(psnr.item())

            table.append(row)

        torch.cuda.empty_cache()
        tbar.update(1)

    df = pd.DataFrame(
        table, columns=('name', 'HR', 'L1', 'L2', 'L3', 'L4', 'L5', 'warp'))
    df = df.sort_values('name')
    df.to_csv(Path(args.weight).parent / 'result.csv', index=False)
