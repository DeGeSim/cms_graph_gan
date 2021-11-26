from dataclasses import dataclass

import torch
from torch.autograd import grad

from fgsim.config import device
from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs
    (https://arxiv.org/abs/1704.00028)
    Args:
        factor (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty
    """

    factor: float = 1.0
    gamma: float = 1.0

    def __call__(self, holder: Holder, real_data, fake_data):
        batch_size = real_data.size(0)

        fake_data = fake_data[:batch_size]

        alpha = torch.rand(batch_size, 1, 1, requires_grad=True).to(device)
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        # compute output of D for interpolated input
        disc_interpolates = holder.disc(interpolates)
        # compute gradients w.r.t the interpolated outputs

        gradients = (
            grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            .contiguous()
            .view(batch_size, -1)
        )

        gradient_penalty = (
            ((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2
        ).mean() * self.factor

        return gradient_penalty
