from typing import List

import torch
from torch_geometric.nn import global_add_pool, global_mean_pool


def aggr_and_sort_points(pc: torch.Tensor):
    # sort by x > y > z > E
    pc_sorted, _ = torch.sort(pc, dim=0, descending=True, stable=True)
    # take the columns relevant for the position
    pc_sorted_pos_slice = pc_sorted[:, :3]
    pc_sorted_energy_slice = pc_sorted[:, 3:4]
    # prepare a list to hold the index of the row in the pc_sorted
    # where pc_sorted[i] ==  pc_sorted[pos_idxs[i]] for the
    # smallest possible pos_idxs[i]
    # This will allow us to aggregate the events at the same point later
    pos_idxs_list: List[int] = []
    comp_row = 0
    for cur_row in range(len(pc)):
        while not torch.all(
            pc_sorted_pos_slice[cur_row] == pc_sorted_pos_slice[comp_row]
        ):
            comp_row += 1
        pos_idxs_list.append(comp_row)
    pos_idxs = torch.tensor(pos_idxs_list, dtype=torch.long, device=pc.device)
    pc_points_aggr = torch.hstack(
        [
            global_mean_pool(pc_sorted_pos_slice, pos_idxs),
            global_add_pool(pc_sorted_energy_slice, pos_idxs),
        ]
    )

    pc_aggr_resorted, _ = torch.sort(
        pc_points_aggr, dim=0, descending=True, stable=True
    )
    return pc_aggr_resorted
