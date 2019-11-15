import torch
import torch.autograd as autograd

from .adversarial_loss import AdversarialLoss
from .perceptual_loss import PerceptualLoss
from .texture_loss import TextureLoss
from .back_projection_loss import BackProjectionLoss
from .metrics import PSNR, SSIM

__all__ = [
    'AdversarialLoss',
    'BackProjectionLoss',
    'PerceptualLoss',
    'TextureLoss',
    'compute_gp',
    'PSNR',
    'SSIM'
]


def compute_gp(netD, real_data, fake_data):
    device = real_data.device
    alpha = torch.rand(real_data.shape[0], 1, 1, 1, device=device)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty
