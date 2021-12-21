import torch

from fgsim.io.queued_dataset import Batch
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, factor: float = 1.0):
        self.factor: float = factor
        self.lossf = torch.nn.MSELoss()

    def __call__(self, holder: Holder, batch: Batch):
        batch_means = torch.mean(batch.pc, (0, 1))
        fake_means = torch.mean(holder.gen_points_w_grad.pc, (0, 1))
        loss = self.factor * self.lossf(fake_means, batch_means)
        loss.backward(retain_graph=True)
        return float(loss)
