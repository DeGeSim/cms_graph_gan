from math import prod

import torch
from torch_scatter import scatter_add

num_z = 45
num_alpha = 16
num_r = 9
dims = [num_z, num_alpha, num_r]

cell_idxs = torch.arange(prod(dims)).reshape(*dims)


# def voxelize(batch, mask, cond):
#     empty = torch.zeros(
#         (batch.shape[0], *dims), dtype=batch.dtype, device=batch.device
#     )
#     # Get the valid hits
#     valid_hits = temp[~mask]
#     valid_coordinates = valid_hits[:, 1:].long().t()
#     shower_index = torch.arange(batch.shape[0]).repeat_interleave(
#         (~mask).float().sum(1).reshape(-1).int()
#     )
#     indices = torch.cat((shower_index.unsqueeze(1), valid_coordinates.t()), dim=1)
#     moritz = torch.arange(batch.shape[0] * dims[0] * dims[1] * dims[2]).reshape(
#         *empty.shape
#     )
#     scatter_index = moritz[
#         indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]
#     ]
#     vox = scatter_add(
#         src=valid_hits[:, 0],
#         index=scatter_index,
#         dim_size=prod(dims) * batch.shape[0],
#     )
#     return vox.reshape(batch.shape[0], *dims)


def sum_dublicate_hits(batch):
    dev = batch.x.device
    batchidx = batch.batch
    n_events = int(batchidx[-1] + 1)

    hitE = batch.x[:, 0]
    pos = batch.x[:, 1:].long()

    cell_idx_per_hit = cell_idxs.to(dev)[pos[:, 0], pos[:, 1], pos[:, 2]]

    # give pos unique values for each event
    eventshift = torch.arange(n_events).to(dev) * prod(dims)
    event_and_cell_idxs = cell_idx_per_hit + eventshift[batchidx]

    # sort the event_and_cell_idxs
    cell_idx_per_hit, index_perm = _scatter_sort(event_and_cell_idxs, batchidx)
    hitE = hitE[index_perm]
    pos = pos[index_perm]
    assert (batchidx[index_perm] == batchidx).all()

    _, invidx, counts = torch.unique_consecutive(
        cell_idx_per_hit, return_inverse=True, return_counts=True
    )

    hitE_new = scatter_add(hitE, invidx)
    sel_new_idx = counts.cumsum(-1) - 1

    batchidx_new = batchidx[sel_new_idx]
    pos_new = pos[sel_new_idx]

    # count the cells, that have been hit multiple times
    n_multihit = scatter_add(counts - 1, batchidx_new)
    new_counts = torch.unique_consecutive(batchidx_new, return_counts=True)[1]

    x_new = torch.hstack([hitE_new.reshape(-1, 1), pos_new])

    # Tests
    # old_counts = torch.unique_consecutive(batchidx, return_counts=True)[1]
    # if "n_pointsv" in batch.keys:
    #     assert (old_counts == batch.n_pointsv).all()
    # assert ((old_counts - new_counts) == n_multihit).all()
    # assert torch.allclose(
    #     scatter_add(hitE_new, batchidx_new), scatter_add(hitE, batchidx)
    # )
    # assert torch.allclose(
    #     scatter_add(pos_new * hitE_new.unsqueeze(-1), batchidx_new, -2),
    #     scatter_add(pos * hitE.unsqueeze(-1), batchidx, -2),
    # )

    batch.n_multihit = n_multihit
    batch.batch = batchidx_new
    batch.x = x_new
    batch.n_pointsv = new_counts
    # need to shift the ptr by the number of removed hits
    batch.ptr[1:] -= n_multihit.cumsum(-1)

    batch.nhits = {
        "n": batch.n_pointsv,
        "n_by_E": batch.n_pointsv / batch.y[:, 0],
    }

    return batch


def _scatter_sort(x, index, dim=-1):
    x, x_perm = torch.sort(x, dim=dim)
    index = index.take_along_dim(x_perm, dim=dim)
    index, index_perm = torch.sort(index, dim=dim, stable=True)
    x = x.take_along_dim(index_perm, dim=dim)
    return x, x_perm.take_along_dim(index_perm, dim=dim)


def test_sum_dublicate_hits():
    from torch_geometric.data import Batch, Data

    batch = Batch.from_data_list(
        [
            Data(
                x=torch.tensor(
                    [
                        [1, 0, 0, 0],
                        [1, 0, 1, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 1],
                    ]
                )
            ),
            Data(
                x=torch.tensor(
                    [
                        [1, 0, 1, 0],
                        [1, 0, 0, 0],
                        [1, 0, 1, 0],
                    ]
                )
            ),
        ]
    )
    batch_new = sum_dublicate_hits(batch.clone())

    batchidx = batch.batch
    batchidx_new = batch_new.batch

    n_multihit = batch_new.n_multihit
    new_counts = batch_new.n_pointsv

    hitE = batch.x[:, 0]
    pos = batch.x[:, 1:].long()
    hitE_new = batch_new.x[:, 0]
    pos_new = batch_new.x[:, 1:].long()

    old_counts = torch.unique_consecutive(batchidx, return_counts=True)[1]
    if "n_pointsv" in batch.keys:
        assert (old_counts == batch.n_pointsv).all()
    assert ((old_counts - new_counts) == n_multihit).all()
    assert torch.allclose(
        scatter_add(hitE_new, batchidx_new), scatter_add(hitE, batchidx)
    )
    assert torch.allclose(
        scatter_add(pos_new * hitE_new.unsqueeze(-1), batchidx_new, -2),
        scatter_add(pos * hitE.unsqueeze(-1), batchidx, -2),
    )
