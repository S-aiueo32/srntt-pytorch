import argparse
from contextlib import contextmanager

from PIL import Image
import torch
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

from models import Swapper, VGG, SRNTT
from losses import PSNR, SSIM

TARGET_LAYERS = ['relu3_1', 'relu2_1', 'relu1_1']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@contextmanager
def timer(name):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    yield
    end.record()

    torch.cuda.synchronize()
    print(f'[{name}] done in {start.elapsed_time(end):.3f} ms')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--ref', '-r', type=str, required=True)
    parser.add_argument('--weight', '-w', type=str, required=True)
    parser.add_argument('--use_weights', action='store_true')
    return parser.parse_args()


def load_image(filename_in, filename_ref):
    img_hr = Image.open(filename_in)
    img_ref = Image.open(filename_ref)

    # adjust to x4
    img_hr = img_hr.resize(
        (x - (x % 4) for x in img_hr.size), Image.BICUBIC)
    img_ref = img_ref.resize(
        (x - (x % 4) for x in img_ref.size), Image.BICUBIC)

    # input image
    img_lr = img_hr.resize(
        (x // 4 for x in img_hr.size), Image.BICUBIC)
    img_bic = img_lr.resize(img_hr.size, Image.BICUBIC)

    # reference image
    img_ref_down = img_ref.resize(
        (x // 4 for x in img_ref.size), Image.BICUBIC)
    img_ref_blur = img_ref_down.resize(img_ref.size, Image.BICUBIC)

    # to tensor
    img_hr = TF.to_tensor(img_hr).unsqueeze(0)
    img_lr = TF.to_tensor(img_lr).unsqueeze(0)
    img_bic = TF.to_tensor(img_bic).unsqueeze(0)
    img_ref = TF.to_tensor(img_ref).unsqueeze(0)
    img_ref_blur = TF.to_tensor(img_ref_blur).unsqueeze(0)

    return {'hr': img_hr, 'lr': img_lr, 'bic': img_bic,
            'ref': img_ref, 'ref_blur': img_ref_blur}


def main(args):
    imgs = load_image(args.input, args.ref)

    vgg = VGG(model_type='vgg19').to(device)
    swapper = Swapper().to(device)

    map_in = vgg(imgs['bic'].to(device), TARGET_LAYERS)
    map_ref = vgg(imgs['ref'].to(device), TARGET_LAYERS)
    map_ref_blur = vgg(imgs['ref_blur'].to(device), TARGET_LAYERS)

    with torch.no_grad(), timer('Feature swapping'):
        maps, weights, correspondences = swapper(map_in, map_ref, map_ref_blur)

    model = SRNTT(use_weights=args.use_weights).to(device)
    model.load_state_dict(torch.load(args.weight))

    img_hr = imgs['hr'].to(device)
    img_lr = imgs['lr'].to(device)
    maps = {
        k: torch.tensor(v).unsqueeze(0).to(device) for k, v in maps.items()}
    weights = torch.tensor(weights).reshape(1, 1, *weights.shape).to(device)

    with torch.no_grad(), timer('Inference'):
        _, img_sr = model(img_lr, maps, weights)

    psnr = PSNR()(img_sr.clamp(0, 1), img_hr.clamp(0, 1)).item()
    ssim = SSIM()(img_sr.clamp(0, 1), img_hr.clamp(0, 1)).item()
    print(f'[Result] PSNR:{psnr:.2f}, SSIM:{ssim:.4f}')

    save_image(img_sr.clamp(0, 1), './out.png')


if __name__ == "__main__":
    main(parse_args())
