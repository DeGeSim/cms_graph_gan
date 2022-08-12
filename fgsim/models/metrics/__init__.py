from typing import Union

import jetnet
import numpy as np
import torch

from fgsim.config import device
from fgsim.io.sel_loader import Batch


def _shape_from_batch(batch: Batch):
    batch_size = int(batch.batch[-1] + 1)
    n_features = int(batch.x.shape[-1])
    assert batch.x.shape[0] % batch_size == 0
    n_points = batch.x.shape[0] // batch_size
    return (batch_size, n_points, n_features)


def w1m(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> Union[float, torch.Tensor, np.float32]:
    shape = _shape_from_batch(gen_batch)
    score = jetnet.evaluation.gen_metrics.w1m(
        jets1=gen_batch.x.reshape(*shape),
        jets2=sim_batch.x.reshape(*shape),
    )[0]
    return min(score * 1e3, 1e8)


def w1p(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> Union[float, torch.Tensor, np.float32]:
    shape = _shape_from_batch(gen_batch)
    score = jetnet.evaluation.gen_metrics.w1p(
        jets1=gen_batch.x.reshape(*shape),
        jets2=sim_batch.x.reshape(*shape),
    )[0]
    return min(score * 1e3, 1e8)


def w1efp(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> Union[float, torch.Tensor, np.float32]:
    shape = _shape_from_batch(gen_batch)
    score = jetnet.evaluation.gen_metrics.w1efp(
        jets1=gen_batch.x.reshape(*shape).cpu(),
        jets2=sim_batch.x.reshape(*shape).cpu(),
        num_batches=1,
        efp_jobs=10,
    )[0]
    return min(score * 1e5, 1e8)


def fpnd(gen_batch: Batch, **kwargs) -> Union[float, torch.Tensor, np.float32]:
    shape = _shape_from_batch(gen_batch)
    try:
        score = jetnet.evaluation.gen_metrics.fpnd(
            jets=gen_batch.x.reshape(*shape),
            jet_type="t",
            batch_size=min(200, shape[0]),
            # device="cpu",
            use_tqdm=False,
        )
        return float(score)
    except ValueError:
        return 1e8
