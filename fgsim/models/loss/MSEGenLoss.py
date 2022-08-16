import torch

from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self) -> None:
        self.lossf = torch.nn.MSELoss()

    def __call__(self, holder: Holder, *args, **kwargs) -> torch.Tensor:
        D_gen = holder.models.disc(holder.gen_points)
        return self.lossf(D_gen, torch.ones_like(D_gen))
