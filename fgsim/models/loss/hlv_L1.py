import torch

from fgsim.io.queued_dataset import Batch
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, var: str, wgrad: bool, factor: float = 1.0, *args, **kwargs):
        self.factor: float = factor
        self.var: str = var
        self.wgrad: bool = wgrad
        self.lossf = torch.nn.L1Loss()

    def __call__(self, holder: Holder, batch: Batch):
        if self.wgrad:
            loss = self.lossf(
                batch.hlvs[self.var], holder.gen_points_w_grad.hlvs[self.var]
            )
        else:
            with torch.no_grad():
                loss = self.lossf(
                    batch.hlvs[self.var], holder.gen_points.hlvs[self.var]
                )

        return self.factor * loss
