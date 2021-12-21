from dataclasses import dataclass

import torch
from torch.autograd import grad

from fgsim.config import conf, device
from fgsim.io.queued_dataset import Batch
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

    def __call__(self, holder: Holder, batch: Batch) -> torch.float:
        real_data = batch.pc
        alpha = torch.rand(conf.loader.batch_size, 1, 1, requires_grad=True).to(
            device
        )
        # randomly mix real and fake data
        interpolates = real_data + alpha * (holder.gen_points.pc - real_data)
        # compute output of D for interpolated input
        disc_interpolates = holder.models.disc(interpolates)
        # compute gradients w.r.t the interpolated outputs

        gradients = (
            grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            .contiguous()
            .view(conf.loader.batch_size, -1)
        )

        gradient_penalty = (
            ((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2
        ).mean()
        loss = self.factor * gradient_penalty
        loss.backward()
        return float(loss)
