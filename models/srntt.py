import torch
import torch.nn as nn
import torch.nn.functional as F

import models


class SRNTT(nn.Module):
    """
    PyTorch Module for SRNTT.
    Now x4 is only supported.

    Parameters
    ---
    ngf : int, optional
        the number of filterd of generator.
    n_blucks : int, optional
        the number of residual blocks for each module.
    """
    def __init__(self, ngf=64, n_blocks=16, use_weights=False):
        super(SRNTT, self).__init__()
        self.content_extractor = ContentExtractor(ngf, n_blocks)
        self.texture_transfer = TextureTransfer(ngf, n_blocks, use_weights)
        models.init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, x, maps, weights=None):
        """
        Parameters
        ---
        x : torch.Tensor
            the input image of SRNTT.
        maps : dict of torch.Tensor
            the swapped feature maps on relu3_1, relu2_1 and relu1_1.
            depths of the maps are 256, 128 and 64 respectively.
        """

        base = F.interpolate(x, None, 4, 'bilinear', False)
        upscale_plain, content_feat = self.content_extractor(x)

        if maps is not None:
            if hasattr(self.texture_transfer, 'a'):  # if weight is used
                upscale_srntt = self.texture_transfer(
                    content_feat, maps, weights)
            else:
                upscale_srntt = self.texture_transfer(
                    content_feat, maps)
            return upscale_plain + base, upscale_srntt + base
        else:
            return upscale_plain + base, None


class ContentExtractor(nn.Module):
    """
    Content Extractor for SRNTT, which outputs maps before-and-after upscale.
    more detail: https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L73.
    Currently this module only supports `scale_factor=4`.

    Parameters
    ---
    ngf : int, optional
        a number of generator's features.
    n_blocks : int, optional
        a number of residual blocks, see also `ResBlock` class.
    """

    def __init__(self, ngf=64, n_blocks=16):
        super(ContentExtractor, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True)
        )
        self.body = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
            # nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf)
        )
        self.tail = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1),
            # nn.Tanh()
        )

    def forward(self, x):
        h = self.head(x)
        h = self.body(h) + h
        upscale = self.tail(h)
        return upscale, h


class TextureTransfer(nn.Module):
    """
    Conditional Texture Transfer for SRNTT,
        see https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L116.
    This module is devided 3 parts for each scales.

    Parameters
    ---
    ngf : int
        a number of generator's filters.
    n_blocks : int, optional
        a number of residual blocks, see also `ResBlock` class.
    """

    def __init__(self, ngf=64, n_blocks=16, use_weights=False):
        super(TextureTransfer, self).__init__()

        # for small scale
        self.head_small = nn.Sequential(
            nn.Conv2d(ngf + 256, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.body_small = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
            # nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf)
        )
        self.tail_small = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        # for medium scale
        self.head_medium = nn.Sequential(
            nn.Conv2d(ngf + 128, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.body_medium = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
            # nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf)
        )
        self.tail_medium = nn.Sequential(
            nn.Conv2d(ngf, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, True),
        )

        # for large scale
        self.head_large = nn.Sequential(
            nn.Conv2d(ngf + 64, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        self.body_large = nn.Sequential(
            *[ResBlock(ngf) for _ in range(n_blocks)],
            # nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(ngf)
        )
        self.tail_large = nn.Sequential(
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(ngf // 2, 3, kernel_size=3, stride=1, padding=1),
            # nn.Tanh()
        )

        if use_weights:
            self.a = nn.Parameter(torch.ones(3), requires_grad=True)
            self.b = nn.Parameter(torch.ones(3), requires_grad=True)

    def forward(self, x, maps, weights=None):
        # compute weighted maps
        if hasattr(self, 'a') and weights is not None:
            for idx, layer in enumerate(['relu3_1', 'relu2_1', 'relu1_1']):
                weights_scaled = F.interpolate(
                    F.pad(weights, (1, 1, 1, 1), mode='replicate'),
                    scale_factor=2**idx,
                    mode='bicubic',
                    align_corners=True) * self.a[idx] + self.b[idx]
                maps[layer] *= torch.sigmoid(weights_scaled)

        # small scale
        h = torch.cat([x, maps['relu3_1']], 1)
        h = self.head_small(h)
        h = self.body_small(h) + x
        x = self.tail_small(h)

        # medium scale
        h = torch.cat([x, maps['relu2_1']], 1)
        h = self.head_medium(h)
        h = self.body_medium(h) + x
        x = self.tail_medium(h)

        # large scale
        h = torch.cat([x, maps['relu1_1']], 1)
        h = self.head_large(h)
        h = self.body_large(h) + x
        x = self.tail_large(h)

        return x


class ResBlock(nn.Module):
    """
    Basic residual block for SRNTT.

    Parameters
    ---
    n_filters : int, optional
        a number of filters.
    """

    def __init__(self, n_filters=64):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(n_filters, n_filters, 3, 1, 1),
        )

    def forward(self, x):
        return self.body(x) + x


if __name__ == "__main__":
    device = torch.device('cuda:0')

    x = torch.rand(16, 3, 24, 24).to(device)

    maps = {}
    maps.update({'relu3_1': torch.rand(16, 256, 24, 24).to(device)})
    maps.update({'relu2_1': torch.rand(16, 128, 48, 48).to(device)})
    maps.update({'relu1_1': torch.rand(16, 64, 96, 96).to(device)})

    model = SRNTT().to(device)
    _, out = model(x, maps)
