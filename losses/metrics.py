import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size

    def forward(self, x, y):
        if x.shape[1] == 3:
            x = kornia.color.rgb_to_grayscale(x)
        if y.shape[1] == 3:
            y = kornia.color.rgb_to_grayscale(y)
        return 1 - kornia.losses.ssim(x, y, self.window_size, 'mean')


class PSNR(nn.Module):
    def __init__(self, max_val=1., mode='Y'):
        super(PSNR, self).__init__()
        self.max_val = max_val
        self.mode = mode

    def forward(self, x, y):
        if self.mode == 'Y' and x.shape[1] == 3 and y.shape[1] == 3:
            x = kornia.color.rgb_to_grayscale(x)
            y = kornia.color.rgb_to_grayscale(y)
        mse = F.mse_loss(x, y, reduction='mean')
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        return psnr
