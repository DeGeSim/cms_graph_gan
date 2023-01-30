import torch
from torch_geometric.data import Batch

from fgsim.config import conf


# Make sure the unchosen particles are close to 0
class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        gen_batch: Batch,
        **kwargs,
    ) -> torch.Tensor:
        assert gen_batch.x.requires_grad
        loss = gen_batch.xnot[..., : conf.loader.x_ftx_energy_pos].abs().sum()
        loss += gen_batch.xnot[..., conf.loader.x_ftx_energy_pos + 1 :].abs().sum()

        # The energy is scaled with a box-cox
        # to move the enery to 0, we need to produce a negative value
        notenergies = gen_batch.xnot[..., conf.loader.x_ftx_energy_pos] + 6
        # it should not be completly outside the distribution -> shift by 6

        loss += (notenergies * (notenergies > 0).float()).sum()

        return loss
