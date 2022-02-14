"""
We create a loss class. This class has a method `__call__`
that takes a `Holder` and a `Batch` as arguments.
It then computes the mean of the points in the batch
and the mean of the points in the generated batch.
It then uses the `torch.nn.MSELoss` to compute the loss
between these two means. It then backpropagates the loss
and returns the loss.
"""
import torch

from fgsim.config import conf
from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, factor: float = 1.0):
        self.factor: float = factor
        self.lossf = torch.nn.MSELoss()

    def __call__(self, holder: Holder, batch: Batch):
        # Aggregate the means of over the points in a event
        sim_means = torch.mean(batch.pc, -2).sort(dim=0).values
        gen_means = torch.mean(holder.gen_points_w_grad.pc, -2).sort(dim=0).values
        assert (
            list(sim_means.shape)
            == list(gen_means.shape)
            == [conf.loader.batch_size, conf.loader.n_features]
        )
        loss = self.factor * self.lossf(gen_means, sim_means)
        loss.backward(retain_graph=True)
        return float(loss)
