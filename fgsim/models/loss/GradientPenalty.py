from dataclasses import dataclass

import torch
from torch.autograd import grad
from torch_geometric.data import Data

from fgsim.config import device
from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs
    (https://arxiv.org/abs/1704.00028)
    Args:
        gamma (float): regularization term of the gradient penalty
    """

    gamma: float = 1.0

    def __call__(self, holder: Holder, batch: Batch) -> torch.float:

        interpol_events = [
            Data(x=interpol_pcs(holder.gen_points[ievent].x, batch[ievent].x))
            for ievent in range(batch.num_graphs)
        ]

        interpol_batch = Batch.from_data_list(interpol_events)

        # compute output of D for interpolated input
        disc_interpolates = holder.models.disc(interpol_batch)
        # compute gradients w.r.t the interpolated outputs
        inputs = [e.x for e in interpol_events]
        grads = grad(
            outputs=disc_interpolates,  # disc output
            inputs=inputs,
            grad_outputs=torch.ones(
                disc_interpolates.size(), device=device
            ),  # I dont understand this
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )
        assert grads[0].shape == inputs[0].shape
        l2s = torch.stack([g.norm(2) for g in grads])

        gradient_penalty = (((l2s - self.gamma) / self.gamma) ** 2).mean()
        loss = gradient_penalty
        return loss


def interpol_pcs(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:

    # The point could very different sizes.
    # So we repeat the smaller one:
    if len(pc1) > len(pc2):
        big_pc = pc1
        small_pc = pc2
    else:
        big_pc = pc2
        small_pc = pc1
    if len(small_pc) == len(big_pc):
        new_pc_len = len(big_pc)
    else:
        new_pc_len = int(torch.randint(len(small_pc), len(big_pc), size=(1,)))

    big_downsample = big_pc[
        torch.multinomial(
            torch.ones(len(big_pc)), num_samples=new_pc_len, replacement=False
        )
    ]
    small_upsample = small_pc[
        torch.multinomial(
            torch.ones(len(small_pc)), num_samples=new_pc_len, replacement=True
        )
    ]

    # randomly interpolate between real and fake data
    alpha = torch.rand(new_pc_len, 1, requires_grad=True).to(device)
    interpolates = small_upsample + alpha * (big_downsample - small_upsample)
    return interpolates
