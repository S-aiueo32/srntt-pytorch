import torch.nn as nn

import models


class Discriminator(nn.Sequential):
    def __init__(self, ndf=32):
        def conv_block(in_channels, out_channels):
            block = [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True),
            ]
            return block

        super(Discriminator, self).__init__(
            *conv_block(3, ndf),
            *conv_block(ndf, ndf * 2),
            *conv_block(ndf * 2, ndf * 4),
            *conv_block(ndf * 4, ndf * 8),
            *conv_block(ndf * 8, ndf * 16),
            nn.Conv2d(ndf * 16, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

        models.init_weights(self, init_type='normal', init_gain=0.02)


class ImageDiscriminator(nn.Sequential):
    def __init__(self, ndf=32):
        def conv_block(in_channels, out_channels):
            block = [
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(out_channels, out_channels, 3, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True),
            ]
            return block

        super(ImageDiscriminator, self).__init__(
            *conv_block(3, ndf),
            *conv_block(ndf, ndf * 2),
            *conv_block(ndf * 2, ndf * 4),
            *conv_block(ndf * 4, ndf * 8),
            *conv_block(ndf * 8, ndf * 16),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 16, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

        models.init_weights(self, init_type='normal', init_gain=0.02)
