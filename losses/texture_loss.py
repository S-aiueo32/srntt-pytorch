import torch
import torch.nn as nn
import torch.nn.functional as F

from models import VGG


def gram_matrix(features):
    N, C, H, W = features.size()
    feat_reshaped = features.view(N, C, -1)

    # Use torch.bmm for batch multiplication of matrices
    gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))

    return gram


class TextureLoss(nn.Module):
    """
    creates a criterion to compute weighted gram loss.
    """
    def __init__(self, use_weights=False):
        super(TextureLoss, self).__init__()
        self.use_weights = use_weights

        self.model = VGG(model_type='vgg19')
        self.register_buffer('a', torch.tensor(-20., requires_grad=False))
        self.register_buffer('b', torch.tensor(.65, requires_grad=False))

    def forward(self, x, maps, weights):
        input_size = x.shape[-1]
        x_feat = self.model(x, ['relu1_1', 'relu2_1', 'relu3_1'])

        if self.use_weights:
            weights = F.pad(weights, (1, 1, 1, 1), mode='replicate')
            for idx, l in enumerate(['relu3_1', 'relu2_1', 'relu1_1']):
                # adjust the scale
                weights_scaled = F.interpolate(
                    weights, None, 2**idx, 'bicubic', True)

                # compute coefficients
                coeff = weights_scaled * self.a.detach() + self.b.detach()
                coeff = torch.sigmoid(coeff)

                # weighting features and swapped maps
                maps[l] = maps[l] * coeff
                x_feat[l] = x_feat[l] * coeff

        # for large scale
        loss_relu1_1 = torch.norm(
            gram_matrix(x_feat['relu1_1']) - gram_matrix(maps['relu1_1']),
        ) / 4. / ((input_size * input_size * 1024) ** 2)

        # for medium scale
        loss_relu2_1 = torch.norm(
            gram_matrix(x_feat['relu2_1']) - gram_matrix(maps['relu2_1'])
        ) / 4. / ((input_size * input_size * 512) ** 2)

        # for small scale
        loss_relu3_1 = torch.norm(
            gram_matrix(x_feat['relu3_1']) - gram_matrix(maps['relu3_1'])
        ) / 4. / ((input_size * input_size * 256) ** 2)

        loss = (loss_relu1_1 + loss_relu2_1 + loss_relu3_1) / 3.

        return loss
