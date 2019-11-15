from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class SwappingDataset(Dataset):
    """
    Dataset class for offline feature swapping.
    """

    def __init__(self,
                 dataroot: Path,
                 input_size: int,
                 scale_factor: int = 4):

        super(SwappingDataset, self).__init__()

        self.input_dir = Path(dataroot) / 'input'
        self.ref_dir = Path(dataroot) / 'ref'

        input_file_len = len(list(self.input_dir.glob('*.png')))
        ref_file_len = len(list(self.ref_dir.glob('*.png')))
        assert input_file_len == ref_file_len,\
            'input/ref folder must have the same files.'

        self.filenames = [f.name for f in self.input_dir.glob('*.png')]

        self.input_size = (input_size, input_size)
        self.output_size = (input_size * 4, input_size * 4)

    def __getitem__(self, index):
        filename = self.filenames[index]

        img_in = Image.open(self.input_dir / filename).convert('RGB')
        img_in_lr = img_in.resize(self.input_size, Image.BICUBIC)
        img_in_up = img_in_lr.resize(self.output_size, Image.BICUBIC)

        img_ref = Image.open(self.ref_dir / filename).convert('RGB')
        img_ref = img_ref.resize(self.output_size, Image.BICUBIC)
        img_ref_lr = img_ref.resize(self.input_size, Image.BICUBIC)
        img_ref_up = img_ref_lr.resize(self.output_size, Image.BICUBIC)

        return {'img_in': TF.to_tensor(img_in_up),
                'img_ref': TF.to_tensor(img_ref),
                'img_ref_blur': TF.to_tensor(img_ref_up),
                'filename': Path(filename).stem}

    def __len__(self):
        return len(self.filenames)
