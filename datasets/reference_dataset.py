from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ReferenceDataset(Dataset):
    """
    Dataset class for Ref-SR.
    """

    def __init__(self,
                 files: list,
                 dataroot: Path,
                 scale_factor: int = 4):

        super(ReferenceDataset, self).__init__()

        self.filenames = files
        self.dataroot = Path(dataroot)
        self.input_dir = self.dataroot / 'input'
        self.ref_dir = self.dataroot / 'ref'
        self.map_dir = self.dataroot / 'map'

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        filename = self.filenames[index]

        img_hr = Image.open(self.input_dir / f'{filename}.png').convert('RGB')
        img_lr = img_hr.resize((x // 4 for x in img_hr.size), Image.BICUBIC)

        img_hr = self.transforms(img_hr)
        img_lr = self.transforms(img_lr)

        with np.load(self.map_dir / f'{filename}.npz') as f:
            relu3_1 = torch.tensor(f['relu3_1'])
            relu2_1 = torch.tensor(f['relu2_1'])
            relu1_1 = torch.tensor(f['relu1_1'])
            weights = torch.tensor(f['weights']).unsqueeze(0)

        # random rotate
        state = random.randint(0, 3)
        img_hr = img_hr.rot90(state, [1, 2])
        img_lr = img_lr.rot90(state, [1, 2])
        relu3_1 = relu3_1.rot90(state, [1, 2])
        relu2_1 = relu2_1.rot90(state, [1, 2])
        relu1_1 = relu1_1.rot90(state, [1, 2])
        weights = weights.rot90(state, [1, 2])

        # random flip
        if random.random() < 0.5:
            img_hr = img_hr.flip([1])
            img_lr = img_lr.flip([1])
            relu3_1 = relu3_1.flip([1])
            relu2_1 = relu2_1.flip([1])
            relu1_1 = relu1_1.flip([1])
            weights = weights.flip([1])
        if random.random() < 0.5:
            img_hr = img_hr.flip([2])
            img_lr = img_lr.flip([2])
            relu3_1 = relu3_1.flip([2])
            relu2_1 = relu2_1.flip([2])
            relu1_1 = relu1_1.flip([2])
            weights = weights.flip([2])

        return {'img_hr': img_hr, 'img_lr': img_lr,
                'maps': {'relu3_1': relu3_1,
                         'relu2_1': relu2_1,
                         'relu1_1': relu1_1},
                'weights': weights}

    def __len__(self):
        return len(self.filenames)


class ReferenceDatasetEval(Dataset):
    """
    Dataset class for Ref-SR.
    """

    def __init__(self,
                 files: list,
                 dataroot: Path,
                 scale_factor: int = 4):

        super(ReferenceDatasetEval, self).__init__()

        self.filenames = files
        self.dataroot = Path(dataroot)
        self.input_dir = self.dataroot / 'input'
        self.ref_dir = self.dataroot / 'ref'
        self.map_dir = self.dataroot / 'map'

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        filename = self.filenames[index]

        img_hr = Image.open(self.input_dir / f'{filename}.png').convert('RGB')
        w, h = img_hr.size
        img_lr = img_hr.resize((w // 4, h // 4), Image.BICUBIC)

        with np.load(self.map_dir / f'{filename}.npz') as f:
            relu3_1 = torch.tensor(f['relu3_1'])
            relu2_1 = torch.tensor(f['relu2_1'])
            relu1_1 = torch.tensor(f['relu1_1'])
            weights = torch.tensor(f['weights']).unsqueeze(0)

        return {'img_hr': self.transforms(img_hr),
                'img_lr': self.transforms(img_lr),
                'maps': {'relu3_1': relu3_1,
                         'relu2_1': relu2_1,
                         'relu1_1': relu1_1},
                'weights': weights}

    def __len__(self):
        return len(self.filenames)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = ReferenceDataset(
        dataroot='/home/ubuntu/srntt-pytorch/data/CUFED'
    )
    dataloader = DataLoader(dataset)

    for batch in dataloader:
        img_hr = batch['img_hr']
        img_lr = batch['img_lr']
        maps = batch['maps']

        break
