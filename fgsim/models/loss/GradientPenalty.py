from dataclasses import dataclass

import torch
from torch.autograd import grad

from fgsim.config import device
from fgsim.ml.holder import Holder
from fgsim.types import Batch, Event, Graph


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
        gen_pc = holder.gen_points.pc
        sim_pc = batch.pc
        # The point could very different sizes.
        # So we repeat the smaller one:
        if len(gen_pc) > len(sim_pc):
            big_pc = gen_pc
            small_pc = sim_pc
        else:
            big_pc = sim_pc
            small_pc = gen_pc
        n_repeats = len(big_pc) // len(small_pc)
        rest = len(big_pc) - n_repeats * len(small_pc)
        rsmall_pc = torch.vstack(
            [small_pc.repeat(n_repeats, 1), small_pc[:rest, :]]
        )
        # Shuffle one of the datasets
        rsmall_pc = rsmall_pc[torch.randperm(rsmall_pc.shape[0])]
        # randomly interpolate between real and fake data
        alpha = torch.rand(len(big_pc), 1, requires_grad=True).to(device)
        interpolates = rsmall_pc + alpha * (big_pc - rsmall_pc)
        interpol_batch = Batch.from_event_list(Event(Graph(x=interpolates)))

        # compute output of D for interpolated input
        disc_interpolates = holder.models.disc(interpol_batch)
        # compute gradients w.r.t the interpolated outputs

        gradients = (
            grad(
                outputs=disc_interpolates,
                inputs=interpolates,
                grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            .contiguous()
            .view(1, -1)
        )

        gradient_penalty = (
            ((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2
        ).mean()
        loss = self.factor * gradient_penalty
        loss.backward()
        return float(loss)
