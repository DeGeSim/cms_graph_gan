from math import prod

import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_add

from fgsim.config import conf, device
from fgsim.io.batch_tools import fix_slice_dict_nodeattr
from fgsim.utils.batch import ptr_from_batchidx

num_z, num_alpha, num_r = 45, 16, 9
# num_z, num_alpha, num_r = 2, 2, 1
dims = [num_z, num_alpha, num_r]


cell_idxs = torch.arange(prod(dims)).reshape(*dims)
full_event_cell_idx = (
    torch.arange(conf.loader.batch_size * dims[0] * dims[1] * dims[2])
    .reshape(conf.loader.batch_size, *dims)
    .to(device)
)


def voxelize(batch):
    batch_size = int(batch.batch[-1] + 1)
    x = batch.x

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
    scatter_index = full_event_cell_idx[
        indices[..., 0], indices[..., 1], indices[..., 2], indices[..., 3]
    ]
    vox = scatter_add(
        src=Ehit,
        index=scatter_index,
        dim_size=prod(dims) * batch_size,
    )
    return vox.reshape(batch_size, *dims)


def invscale_add_position(batch):
    from .objcol import scaler

    if batch.pos is not None:
        if len(batch.x) == len(batch.pos):
            return
    pos_l = []
    for iftx in range(batch.x.shape[1]):
        if iftx == conf.loader.x_ftx_energy_pos:
            continue
        pos_l.append(
            torch.tensor(
                scaler.transfs_x[iftx].inverse_transform(
                    batch.x[:, [iftx]].detach().cpu().double().numpy()
                )
            ).to(batch.x.device)
        )
    batch.pos = torch.hstack(pos_l).long()


def cell_occ_per_hit(batch):
    batch_size = int(batch.batch[-1] + 1)
    dev = batch.x.device
    fulldim = (batch_size, *dims)

    invscale_add_position(batch)
    indices = torch.hstack((batch.batch.unsqueeze(1), batch.pos))
    scatter_index = full_event_cell_idx[
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


def __globalidx_from_pos(pos, batchidx):
    pos_z, pos_alpha, pos_r = pos.T
    assert (pos >= 0).all()
    assert (pos_z < num_z).all()
    assert (pos_alpha < num_alpha).all()
    assert (pos_r < num_r).all()
    dev = pos.device
    n_events = int(batchidx[-1] + 1)
    # create and index that indexes each cell for each event
    globalidx = cell_idxs.to(dev)[pos_z, pos_alpha, pos_r]
    # add the shiftvector to diffentiate for each event
    eventshift = torch.arange(n_events).to(dev) * prod(dims)
    globalidx = globalidx + eventshift[batchidx]
    return globalidx, eventshift


def sum_dublicate_hits(batch, forbid_dublicates=False, shiftmultihit=True):
    old_dev = batch.x.device
    # batch = batch.to("cpu")

    batchidx = batch.batch
    assert (batchidx.diff() >= 0).all()
    globalidx, eventshift = __globalidx_from_pos(batch.x[:, 1:].long(), batchidx)

    if shiftmultihit:
        # get new positions and global index
        # and the current index of these events
        batch, globalidx = _move_doublehits_to_neighbor_cells(
            batch,
            globalidx.clone(),
            eventshift.clone(),
        )

    # sort the globalidx
    globalidx, index_perm = _scatter_sort(globalidx, batchidx)

    hitE = batch.x[:, 0]
    pos = batch.x[:, 1:].long()
    hitE = hitE[index_perm]
    pos = pos[index_perm]
    assert (batchidx[index_perm] == batchidx).all()

    # unique_cells_idx counts up every time a new cell in an
    # even is accessed in globalidx
    # counts gives the times the cell/event idx is occupied
    _, unique_cells_idx, counts = torch.unique(
        globalidx, return_inverse=True, return_counts=True
    )

    if forbid_dublicates:
        assert (counts - 1 == 0).all()

    # begin sum
    hitE_new = scatter_add(hitE, unique_cells_idx)
    sel_new_idx = counts.cumsum(-1) - 1
    if forbid_dublicates:
        assert (sel_new_idx == torch.arange(len(batch.x))).all()

    batchidx_new = batchidx[sel_new_idx]
    pos_new = pos[sel_new_idx]

    # count the cells, that have been hit multiple times
    n_multihit = scatter_add(counts - 1, batchidx_new)
    if forbid_dublicates:
        assert (n_multihit == 0).all()
    new_counts = torch.unique_consecutive(batchidx_new, return_counts=True)[1]

    x_new = torch.hstack([hitE_new.reshape(-1, 1), pos_new])

    # TODO remove sanity test:
    old_counts = torch.unique_consecutive(batchidx, return_counts=True)[1]
    if "n_pointsv" in batch.keys:
        assert (old_counts == batch.n_pointsv).all()
    assert ((old_counts - new_counts) == n_multihit).all()
    assert torch.allclose(
        scatter_add(hitE_new, batchidx_new), scatter_add(hitE, batchidx)
    )
    if not shiftmultihit:
        assert torch.allclose(
            scatter_add(pos_new * hitE_new.unsqueeze(-1), batchidx_new, -2),
            scatter_add(pos * hitE.unsqueeze(-1), batchidx, -2),
        )

    if forbid_dublicates:
        assert (n_multihit == 0).all()
        assert (batch.batch == batchidx_new).all()
        assert (batch.n_pointsv == new_counts).all()
        for i in range(4):
            assert torch.allclose(batch.x.T[i][index_perm], x_new.T[i])

    batch.n_multihit = n_multihit
    batch.batch = batchidx_new
    batch.x = x_new
    batch.n_pointsv = new_counts
    # need to shift the ptr by the number of removed hits
    batch.ptr = ptr_from_batchidx(batchidx_new)

    batch.nhits = {
        "n": batch.n_pointsv,
        "n_by_E": batch.n_pointsv / batch.y[:, 0],
    }
    fix_slice_dict_nodeattr(batch, "x")

    return batch.to(old_dev)


def _move_doublehits_to_neighbor_cells(
    batch: Batch, globalidx: torch.Tensor, eventshift: torch.Tensor
):
    pos: torch.Tensor = batch.x[:, 1:].long()
    batchidx: torch.Tensor = batch.batch

    dev = pos.device

    _, unique_cells_idx, counts = torch.unique(
        globalidx, return_inverse=True, return_counts=True
    )
    # select the second hit in each cell for moving around
    # TODO change to lowest eneergy hit
    mhit_idxs = (
        torch.cat([torch.tensor([0]), 1 - unique_cells_idx.diff()])
        .nonzero()
        .squeeze()
    )
    shift_options = len(dims) * 2
    # start value for each shift
    shift_state = torch.randint_like(mhit_idxs, 0, shift_options)
    idxs_to_overwrite = []
    new_pos_list = []
    new_global_list = []

    for _ in range(shift_options):
        if len(mhit_idxs) == 0:
            break
        new_pos = _shift_pos(pos[mhit_idxs], shift_state)
        # get the new double index for the shifted hits
        new_global = (
            cell_idxs.to(dev)[
                new_pos.T[0],
                new_pos.T[1],
                new_pos.T[2],
            ]
            + eventshift[batchidx[mhit_idxs]]
        )
        # The new position is valid iff it's A) not occupied
        new_pos_is_free = torch.isin(
            new_global, torch.cat([globalidx, *new_global_list])
        )

        # and B) this is the first hit to be shifted to this cell
        # we now need and index that filters mhit_idxs
        # and only lets the first though.
        # (4,3,4,1,3)   -> (1,1,0,1,0) or (0,1,3)
        # we can check if the index  of new_global.unique is increaseing
        # but for this we need to sort the indexes again
        new_global, perm = new_global.sort()
        mhit_idxs = mhit_idxs[perm]
        new_pos = new_pos[perm]

        # now we can check of the unique rev index is increasing
        # which tells us of the cell is being accessed for the first time
        first_new_hit = torch.cat(
            (
                torch.tensor([1]),
                new_global.unique(return_inverse=True)[1].diff(),
            )
        ).bool()

        # combine the two
        valide_shift_index = new_pos_is_free & first_new_hit

        # apppend the shifted hits to the list
        if valide_shift_index.sum():
            idxs_to_overwrite.append(mhit_idxs[valide_shift_index])
            new_pos_list.append(new_pos[valide_shift_index])
            new_global_list.append(new_global[valide_shift_index])
        # rerun loop with the remaining multihits
        mhit_idxs = mhit_idxs[~valide_shift_index]
        shift_state = (shift_state[~valide_shift_index] + 1) % shift_options

    if len(idxs_to_overwrite):
        # overwrite old position and global index
        stacked_idxs_to_overwrite = torch.cat(idxs_to_overwrite)
        globalidx[stacked_idxs_to_overwrite] = torch.cat(new_global_list)
        pos[stacked_idxs_to_overwrite] = torch.cat(new_pos_list)

        batch.x[:, 1:] = pos
    return batch, globalidx


def _scatter_sort(x, index, dim=-1):
    x, x_perm = torch.sort(x, dim=dim)
    index = index.take_along_dim(x_perm, dim=dim)
    index, index_perm = torch.sort(index, dim=dim, stable=True)
    x = x.take_along_dim(index_perm, dim=dim)
    return x, x_perm.take_along_dim(index_perm, dim=dim)


def _shift_pos(pos, shift_state):
    pos = pos.clone()
    directon = (shift_state % 2) * 2 - 1
    dim = shift_state // 2
    pos[torch.arange(len(pos)), dim] += directon

    # rotate alpha
    alphas = pos[:, 1]
    alphas[alphas > num_alpha - 1] -= num_alpha
    alphas[alphas < 0] -= num_alpha
    pos[:, 1] = alphas

    # clamp r and z
    pos = torch.clamp(pos, torch.tensor([0, 0, 0]), torch.tensor(dims) - 1)
    return pos


def test_sum_dublicate_hits():
    from torch_geometric.data import Batch, Data

    batch = Batch.from_data_list(
        [
            Data(
                x=torch.tensor(
                    [
                        [1, 0, 0, 0],
                    ]
                ),
                y=torch.tensor([[1, 1]]),
                # n_pointsv=torch.tensor(3),
            ),
            Data(
                x=torch.tensor(
                    [
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 1, 1, 0],
                        [1, 1, 1, 0],
                        # [1, 1, 0, 0],
                        # [1, 1, 0, 0],
                        # [1, 0, 0, 0],
                        # [1, num_z - 1, num_alpha - 1, num_r - 1],
                    ]
                ),
                y=torch.tensor([[1, 1]]),
                # n_pointsv=torch.tensor(4),
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


if __name__ == "__main__":
    test_sum_dublicate_hits()
