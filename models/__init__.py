import torch.nn.init as init

from .discriminator import Discriminator, ImageDiscriminator
from .srntt import SRNTT, ContentExtractor
from .swapper import Swapper
from .vgg import VGG

__all__ = [
    'Discriminator',
    'ImageDiscriminator',
    'SRNTT',
    'ContentExtractor',
    'Swapper',
    'VGG'
]


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        name = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in name or 'Linear' in name):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in name:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
