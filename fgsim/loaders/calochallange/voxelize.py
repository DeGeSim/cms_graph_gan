from math import prod

import torch
from torch_scatter import scatter_add

from fgsim.config import conf
from fgsim.io.batch_tools import fix_slice_dict_nodeattr
from fgsim.io.sel_loader import scaler

num_z = 45
num_alpha = 16
num_r = 9
dims = [num_z, num_alpha, num_r]

cell_idxs = torch.arange(prod(dims)).reshape(*dims)


def voxelize(batch):
    batch_size = int(batch.batch[-1] + 1)
    x = batch.x
    empty = torch.zeros((batch_size, *dims), dtype=x.dtype, device=x.device)
    # Get the valid hits
    # valid_hits = temp[~mask]
    # Ehit = valid_hits[:, 0]
    # valid_coordinates = valid_hits[:, 1:].long().t()
    # shower_index = torch.arange(batch_size).repeat_interleave(
    #     (~mask).float().sum(1).reshape(-1).int()
    # )
    shower_index = batch.batch
    Ehit = x.T[0]
    valid_coordinates = x.T[1:].int()
    indices = torch.cat((shower_index.unsqueeze(1), valid_coordinates.t()), dim=1)
    moritz = torch.arange(batch_size * dims[0] * dims[1] * dims[2]).reshape(
        *empty.shape
    )
    scatter_index = moritz[
        indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]
    ]
    vox = scatter_add(
        src=Ehit,
        index=scatter_index,
        dim_size=prod(dims) * batch_size,
    )
    return vox.reshape(batch_size, *dims)


def get_pos(batch):
    pos_l = []
    for iftx in range(batch.x.shape[1]):
        if iftx == conf.loader.x_ftx_energy_pos:
            continue
        pos_l.append(
            torch.tensor(
                scaler.transfs_x[iftx].inverse_transform(
                    batch.x[:, [iftx]].detach().cpu().numpy()
                )
            ).to(batch.x.device)
        )
    return torch.hstack(pos_l)


def cell_occ_per_hit(batch):
    batch_size = int(batch.batch[-1] + 1)
    dev = batch.x.device
    x = batch.x.detach()
    fulldim = (batch_size, *dims)
    empty = torch.zeros(fulldim, dtype=torch.int, device=x.device)

    valid_coordinates = get_pos(batch).int()
    indices = torch.hstack((batch.batch.unsqueeze(1), valid_coordinates))
    moritz = (
        torch.arange(batch_size * dims[0] * dims[1] * dims[2])
        .reshape(*empty.shape)
        .to(dev)
    )
    scatter_index = moritz[
        indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]
    ]
    occ = scatter_add(
        src=torch.ones(len(batch.batch), dtype=torch.int, device=dev),
        index=scatter_index,
        dim_size=prod(dims) * batch_size,
    ).reshape(*fulldim)
    batch_occ = occ[
        indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]
    ]
    return batch_occ


def sum_dublicate_hits(batch):
    old_dev = batch.x.device
    batch = batch.to("cpu")
    dev = batch.x.device
    batchidx = batch.batch
    n_events = int(batchidx[-1] + 1)

    hitE = batch.x[:, 0]
    pos = batch.x[:, 1:].long()

    pos_z, pos_alpha, pos_r = pos.T
    assert (pos >= 0).all()
    assert (pos_z < num_z).all()
    assert (pos_alpha < num_alpha).all()
    assert (pos_r < num_r).all()

    cell_idx_per_hit = cell_idxs.to(dev)[pos_z, pos_alpha, pos_r]

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

    # # Tests
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
    fix_slice_dict_nodeattr(batch, "x")

    return batch.to(old_dev)


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
                        [1, num_z - 1, num_alpha - 1, num_r - 1],
                    ]
                ),
                y=torch.tensor([[1, 1]]),
                n_pointsv=torch.tensor(4),
            ),
            Data(
                x=torch.tensor(
                    [
                        [1, 0, 1, 0],
                        [1, 0, 0, 0],
                        [1, 0, 1, 0],
                    ]
                ),
                y=torch.tensor([[1, 1]]),
                n_pointsv=torch.tensor(3),
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


test_sum_dublicate_hits()
