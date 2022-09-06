from typing import Union

import jetnet
import numpy as np
import torch

from fgsim.io.sel_loader import Batch
from fgsim.utils.jetnetutils import to_stacked_mask


def w1m(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> Union[float, torch.Tensor, np.float32]:
    score = jetnet.evaluation.gen_metrics.w1m(
        jets1=to_stacked_mask(gen_batch)[:1000, ..., :3],
        jets2=to_stacked_mask(sim_batch)[:1000, ..., :3],
    )[0]
    return min(score * 1e3, 1e8)


def w1p(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> Union[float, torch.Tensor, np.float32]:
    score = jetnet.evaluation.gen_metrics.w1p(
        jets1=to_stacked_mask(gen_batch)[:1000, ..., :3],
        jets2=to_stacked_mask(sim_batch)[:1000, ..., :3],
    )[0]
    return min(score * 1e3, 1e8)


def w1efp(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> Union[float, torch.Tensor, np.float32]:
    score = jetnet.evaluation.gen_metrics.w1efp(
        jets1=to_stacked_mask(gen_batch)[:1000, ..., :3].cpu(),
        jets2=to_stacked_mask(sim_batch)[:1000, ..., :3].cpu(),
        num_batches=1,
        efp_jobs=10,
    )[0]
    return min(score * 1e5, 1e8)


def fpnd(gen_batch: Batch, **kwargs) -> Union[float, torch.Tensor, np.float32]:
    try:
        score = jetnet.evaluation.gen_metrics.fpnd(
            jets=to_stacked_mask(gen_batch)[:1000, ..., :3],
            jet_type="t",
            use_tqdm=False,
        )
        return float(score)
    except ValueError:
        return 1e8
