from typing import Union

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch


def _shape_from_batch(batch: Union[Data, Batch]):
    # batch_size = int(batch.batch[-1] + 1)
    # n_features = int(batch.x.shape[-1])
    # assert batch.x.shape[0] % batch_size == 0
    # n_points = batch.x.shape[0] // batch_size
    return (-1, 30, 3)


def _to_stacked_mask(batch: Union[Data, Batch]):
    x, mask = to_dense_batch(x=batch.x, batch=batch.batch, max_num_nodes=30)
    return torch.cat((x, mask.float().reshape(-1, 30, 1)), dim=2)


def _pp_for_jetnet_metric(batch: Union[Data, Batch]) -> torch.Tensor:
    shape = list(_shape_from_batch(batch))
    x = _to_stacked_mask(batch)
    shape[-1] += 1  # increse for the mask
    return x.reshape(*shape)
