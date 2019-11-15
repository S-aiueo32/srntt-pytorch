import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.swapping_dataset import SwappingDataset
from models import VGG
from models.swapper import Swapper

TARGET_LAYERS = ['relu3_1', 'relu2_1', 'relu1_1']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--patch_size', default=3)
    parser.add_argument('--stride', default=1)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main(args):
    dataroot = Path(args.dataroot)
    save_dir = dataroot / 'map'
    save_dir.mkdir(exist_ok=True)

    dataset = SwappingDataset(
        dataroot=dataroot, input_size=40 if 'CUFED' in dataroot.name else 80)
    dataloader = DataLoader(dataset)
    model = VGG(model_type='vgg19').to(device)
    swapper = Swapper(args.patch_size, args.stride).to(device)

    for i, batch in enumerate(tqdm(dataloader), 1):
        img_in = batch['img_in'].to(device)
        img_ref = batch['img_ref'].to(device)
        img_ref_blur = batch['img_ref_blur'].to(device)

        map_in = model(img_in, TARGET_LAYERS)
        map_ref = model(img_ref, TARGET_LAYERS)
        map_ref_blur = model(img_ref_blur, TARGET_LAYERS)

        maps, weights, correspondences = swapper(map_in, map_ref, map_ref_blur)

        np.savez_compressed(save_dir / f'{batch["filename"][0]}.npz',
                            relu1_1=maps['relu1_1'],
                            relu2_1=maps['relu2_1'],
                            relu3_1=maps['relu3_1'],
                            weights=weights,
                            correspondences=correspondences)

        if args.debug and i == 10:
            break


if __name__ == "__main__":
    main(parse_args())
