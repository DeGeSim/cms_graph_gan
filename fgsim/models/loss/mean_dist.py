import torch

from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, factor: float = 1.0):
        self.factor: float = factor
        self.lossf = torch.nn.MSELoss()

    def __call__(self, holder: Holder, batch: torch.Tensor):
        batch_means = torch.mean(batch, (0, 1))
        fake_means = torch.mean(holder.gen_points_w_grad, (0, 1))
        loss = self.lossf(fake_means, batch_means)
        return self.factor * loss
