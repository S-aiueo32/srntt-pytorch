import torch
import torch.nn as nn
import torch.nn.functional as F

from models import VGG


class PerceptualLoss(nn.Module):
    """
    PyTorch module for perceptual loss.

    Parameters
    ---
    model_type : str
        select from [`vgg11`, `vgg11bn`, `vgg13`, `vgg13bn`,
                     `vgg16`, `vgg16bn`, `vgg19`, `vgg19bn`, ].
    target_layers : str
        the layer name you want to compare.
    norm_type : str
        the type of norm, select from ['mse', 'fro']
    """
    def __init__(self,
                 model_type: str = 'vgg19',
                 target_layer: str = 'relu5_1',
                 norm_type: str = 'fro'):
        super(PerceptualLoss, self).__init__()

        assert norm_type in ['mse', 'fro']

        self.model = VGG(model_type=model_type)
        self.target_layer = target_layer
        self.norm_type = norm_type

    def forward(self, x, y):
        x_feat, *_ = self.model(x, [self.target_layer]).values()
        y_feat, *_ = self.model(y, [self.target_layer]).values()

        # frobenius norm in the paper, but mse loss is actually used in
        # https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L376.
        if self.norm_type == 'mse':
            loss = F.mse_loss(x_feat, y_feat)
        elif self.norm_type == 'fro':
            loss = torch.norm(x_feat - y_feat, p='fro')

        return loss
