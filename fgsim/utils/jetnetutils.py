from typing import Union

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from fgsim.config import conf


def to_stacked_mask(batch: Union[Data, Batch]):
    assert conf.loader.n_points * (batch.batch[-1] + 1) > len(batch.x)
    x, mask = to_dense_batch(
        x=batch.x, batch=batch.batch, max_num_nodes=conf.loader.n_points
    )
    return torch.cat((x, mask.float().reshape(-1, conf.loader.n_points, 1)), dim=2)
