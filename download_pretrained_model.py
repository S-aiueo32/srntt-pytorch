import argparse
from collections import OrderedDict
from pathlib import Path

from google_drive_downloader import GoogleDriveDownloader as gdd
from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from models import SRNTT

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./weights')
    parser.add_argument('--save_image', action='store_true')
    return parser.parse_args()


def main(args):
    save_dir = Path(args.save_dir)
    output_path = save_dir / 'SRGAN.pth'

    # download pre-trained SRGAN model of MMSR,
    # see also: https://github.com/open-mmlab/mmsr
    gdd.download_file_from_google_drive(
        file_id='1c0YNygNMfTLynR-C3y7nsZgaWbczbW5j',
        dest_path=output_path
    )

    # load state_dict
    old_state_dict = torch.load(output_path)

    # create a new state_dict with renamed keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        if 'conv_first' in new_key:
            new_key = new_key.replace('conv_first', 'head.0')
        if 'recon_trunk' in new_key:
            new_key = new_key.replace('recon_trunk', 'body')
        if '.conv1.weight' in new_key:
            new_key = new_key.replace('.conv1.weight', '.body.0.weight')
        if '.conv1.bias' in new_key:
            new_key = new_key.replace('.conv1.bias', '.body.0.bias')
        if '.conv2.weight' in new_key:
            new_key = new_key.replace('.conv2.weight', '.body.2.weight')
        if '.conv2.bias' in new_key:
            new_key = new_key.replace('.conv2.bias', '.body.2.bias')
        if 'upconv1' in new_key:
            new_key = new_key.replace('upconv1', 'tail.0')
        if 'upconv2' in new_key:
            new_key = new_key.replace('upconv2', 'tail.3')
        if 'HRconv' in new_key:
            new_key = new_key.replace('HRconv', 'tail.6')
        if 'conv_last' in new_key:
            new_key = new_key.replace('conv_last', 'tail.8')
        new_state_dict[new_key] = val

    # check the loading and forwarding
    model = SRNTT().to(device)
    model.content_extractor.load_state_dict(new_state_dict)
    print('Loading succeeded.')

    img = Image.open('./data/CUFED5/000_0.png').convert('RGB')
    img = TF.to_tensor(img).to(device)
    img = img.unsqueeze(0)
    out, _ = model(img, None)
    print('Forwarding succeeded.')

    if args.save_image:
        save_image(out.clamp(0, 1), save_dir / 'tmp.png')
        print('Please verify the output image.')

    # save the new state_dict
    torch.save(new_state_dict, output_path)


if __name__ == "__main__":
    main(parse_args())
