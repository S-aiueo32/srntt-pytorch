import torch.nn as nn
import torch.nn.functional as F


class BackProjectionLoss(nn.Module):
    def __init__(self, scale_factor=4):
        super(BackProjectionLoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x, y):
        assert x.shape[2] == y.shape[2] * self.scale_factor
        assert x.shape[3] == y.shape[3] * self.scale_factor
        x = F.interpolate(x, y.size()[-2:], mode='bicubic', align_corners=True)
        return F.l1_loss(x, y)
