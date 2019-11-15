from pathlib import Path
import functools

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image(path: Path):
    return path.suffix in IMG_EXTENSIONS


def pad(img, scale):
    width, height = img.size
    pad_h = width % scale
    pad_v = height % scale
    img = TF.pad(
        img, (0, 0, scale - pad_h, scale - pad_v), padding_mode='reflect')
    return img


class BasicDataset(Dataset):
    def __init__(self, data_dir, scale_factor, patch_size=0, mode='train'):

        assert patch_size % scale_factor == 0
        assert (mode == 'train' and patch_size != 0) or mode == 'eval'

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)

        self.filenames = [f for f in data_dir.glob('*') if is_image(f)]
        self.scale_factor = scale_factor

        if mode == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomCrop(
                    patch_size, pad_if_needed=True, padding_mode='reflect'),
                transforms.RandomApply([
                    functools.partial(TF.rotate, angle=0),
                    functools.partial(TF.rotate, angle=90),
                    functools.partial(TF.rotate, angle=180),
                    functools.partial(TF.rotate, angle=270),
                ]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
        elif mode == 'eval':
            self.filenames.sort()
            if patch_size > 0:
                self.transforms = transforms.Compose([
                    transforms.CenterCrop(patch_size)
                ])
            else:
                self.transforms = transforms.Compose([
                    functools.partial(pad, scale=scale_factor)
                ])
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = Image.open(filename).convert('RGB')
        img_hr = self.transforms(img)
        down_size = [l // self.scale_factor for l in img_hr.size[::-1]]
        img_lr = TF.resize(img_hr, down_size, interpolation=Image.BICUBIC)
        return {'lr': TF.to_tensor(img_lr) * 2 - 1,
                'hr': TF.to_tensor(img_hr) * 2 - 1,
                'path': filename.stem}

    def __len__(self):
        return len(self.filenames)
