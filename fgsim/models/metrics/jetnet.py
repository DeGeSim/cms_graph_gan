from typing import Union

import jetnet
import numpy as np
import torch
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.utils.jetnetutils import to_stacked_mask

jet_type = conf.loader.dataset_glob.strip("**/").strip(".hdf5")


def w1m(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> Union[float, torch.Tensor, np.float32]:
    score = jetnet.evaluation.gen_metrics.w1m(
        jets1=to_stacked_mask(gen_batch)[:10000, ..., :3],
        jets2=to_stacked_mask(sim_batch)[:10000, ..., :3],
    )[0]
    return min(score * 1e3, 1e5)


def w1p(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> Union[float, torch.Tensor, np.float32]:
    score = jetnet.evaluation.gen_metrics.w1p(
        jets1=to_stacked_mask(gen_batch)[:10000, ..., :3],
        jets2=to_stacked_mask(sim_batch)[:10000, ..., :3],
    )[0]
    return min(score * 1e3, 1e5)


def w1efp(
    gen_batch: Batch, sim_batch: Batch, **kwargs
) -> Union[float, torch.Tensor, np.float32]:
    score = jetnet.evaluation.gen_metrics.w1efp(
        jets1=to_stacked_mask(gen_batch)[:10000, ..., :3].cpu(),
        jets2=to_stacked_mask(sim_batch)[:10000, ..., :3].cpu(),
        num_batches=1,
        efp_jobs=10,
    )[0]
    return min(score * 1e5, 1e5)


def fpnd(gen_batch: Batch, **kwargs) -> Union[float, torch.Tensor, np.float32]:
    try:
        score = jetnet.evaluation.gen_metrics.fpnd(
            jets=to_stacked_mask(gen_batch)[:50000, ..., :3],
            jet_type=jet_type,
            use_tqdm=False,
        )
        return min(float(score), 1e5)
    except ValueError:
        return 1e5
