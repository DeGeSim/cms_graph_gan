import torch

from fgsim.io.sel_loader import Batch
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self) -> None:
        self.lossf = torch.nn.MSELoss()

    def __call__(self, holder: Holder, batch: Batch) -> torch.Tensor:
        D_sim = holder.models.disc(batch)
        D_gen = holder.models.disc(holder.gen_points)
        return self.lossf(torch.ones_like(D_sim), D_sim) + self.lossf(
            torch.zeros_like(D_gen), D_gen
        )
