from typing import List

import torch

from fgsim.config import conf
from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder

from .mmd import MMD


class LossGen:
    def __init__(self, factor: float, kernel, bandwidth: List[float]) -> None:
        self.factor = factor
        self.kernel = kernel
        self.bandwidth = bandwidth

    def __call__(self, holder: Holder, batch: Batch, *args, **kwargs):
        shape = (
            conf.loader.batch_size,
            conf.loader.max_points,
            conf.loader.n_features,
        )
        sim_sample = batch.x.reshape(*shape)
        gen_sample = holder.gen_points_w_grad.x.reshape(*shape)

        losses: List[torch.Tensor] = []
        for ifeature in range(conf.loader.n_features):
            losses.append(
                MMD(
                    sort_by_feature(sim_sample, ifeature),
                    sort_by_feature(gen_sample, ifeature),
                    bandwidth=self.bandwidth,
                    kernel=self.kernel,
                )
            )
        loss: torch.Tensor = self.factor * sum(losses)
        loss.backward(retain_graph=True)
        return float(loss)


def sort_by_feature(batch: torch.Tensor, ifeature: int) -> torch.Tensor:
    assert 0 <= ifeature <= batch.shape[-1]
    sorted_ftx_idxs = torch.argsort(batch[..., ifeature]).reshape(-1)
    batch_idxs = (
        torch.arange(batch.shape[0]).repeat_interleave(batch.shape[1]).reshape(-1)
    )
    batch_sorted = batch[batch_idxs, sorted_ftx_idxs, :].reshape(*batch.shape)
    return batch_sorted
