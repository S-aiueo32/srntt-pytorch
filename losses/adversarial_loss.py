import torch
import torch.nn as nn


class AdversarialLoss(nn.Module):
    """
    PyTorch module for GAN loss.
    This code is inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
    """
    def __init__(self,
                 gan_mode='wgangp',
                 target_real_label=1.0,
                 target_fake_label=0.0):

        super(AdversarialLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(f'gan mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).detach()

    def forward(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = - prediction.mean()
            else:
                loss = prediction.mean()
        return loss
