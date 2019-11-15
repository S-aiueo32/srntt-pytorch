from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg

__all__ = ['VGG']

NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1',
        'conv2_1', 'relu2_1', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5',
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5',
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5',
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
    ]
}


def insert_bn(names: list):
    """
    Inserts bn layer after each conv.

    Parameters
    ---
    names : list
        The list of layer names.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            pos = name.replace('conv', '')
            names_bn.append('bn' + pos)
    return names_bn


class VGG(nn.Module):
    """
    Creates any type of VGG models.

    Parameters
    ---
    model_type : str
        The model type you want to load.
    requires_grad : bool, optional
        Whethere compute gradients.
    """
    def __init__(self, model_type: str, requires_grad: bool = False):
        super(VGG, self).__init__()

        features = getattr(vgg, model_type)(True).features
        self.names = NAMES[model_type.replace('_bn', '')]
        if 'bn' in model_type:
            self.names = insert_bn(self.names)

        self.net = nn.Sequential(OrderedDict([
            (k, v) for k, v in zip(self.names, features)
        ]))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer(
            name='vgg_mean',
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]],
                                requires_grad=False)
        )
        self.register_buffer(
            name='vgg_std',
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]],
                                requires_grad=False)
        )

    def z_score(self, x):
        x = x.sub(self.vgg_mean.detach())
        x = x.div(self.vgg_std.detach())
        return x

    def forward(self, x: torch.Tensor, targets: list) -> dict:
        """
        Parameters
        ---
        x : torch.Tensor
            The input tensor normalized to [0, 1].
        target : list of str
            The layer names you want to pick up.
        Returns
        ---
        out_dict : dict of torch.Tensor
            The dictionary of tensors you specified.
            The elements are ordered by the original VGG order. 
        """

        assert all([t in self.names for t in targets]),\
            'Specified name does not exist.'

        if torch.all(x < 0.) and torch.all(x > 1.):
            warnings.warn('input tensor is not normalize to [0, 1].')

        x = self.z_score(x)

        out_dict = OrderedDict()
        for key, layer in self.net._modules.items():
            x = layer(x)
            if key in targets:
                out_dict.update({key: x})
            if len(out_dict) == len(targets):  # to reduce wasting computation
                break

        return out_dict
