from typing import Dict, List

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_mean_pool

from fgsim.config import conf
from fgsim.monitoring.logger import logger

# @classmethod
# def from_event_list(cls, *events: Event):
#     graph = DataBatch.from_data_list([event.graph for event in events])
#     hlvs = {
#         key: torch.stack([event.hlvs[key] for event in events])
#         for key in events[0].hlvs
#     }
#     return cls(graph=graph, hlvs=hlvs)


def batch_from_pcs_list(pcs: torch.Tensor, events: torch.Tensor) -> Batch:
    pcs_list = [pcs[events == ievent] for ievent in range(max(events) + 1)]
    event_list = [Data(x=pc) for pc in pcs_list]

    return Batch.from_data_list(event_list)


def batch_compute_hlvs(batch: Batch) -> Batch:
    event_list = [x for x in batch.to_data_list()]
    for event in event_list:
        event.compute_hlvs()
    batch = Batch.from_data_list(event_list)
    return batch


def compute_hlvs(graph: Data) -> Dict[str, torch.Tensor]:
    hlvs: Dict[str, torch.Tensor] = {}
    hlvs["energy_sum"] = torch.sum(graph.x[:, 0])
    hlvs["energy_sum_std"] = torch.std(graph.x[:, 0])
    e_weight = graph.x[:, 0] / hlvs["energy_sum"]
    # if torch.isnan(torch.sum(e_weight)):
    #     logger.warning(f"energy: vec {min_mean_max(graph.x[:, 0])}")
    #     logger.warning(f"x: vec {min_mean_max(graph.x[:, 1])}")
    #     logger.warning(f"y: vec {min_mean_max(graph.x[:, 2])}")
    #     logger.warning(f"z: vec {min_mean_max(graph.x[:, 3])}")

    for irow, key in enumerate(conf.loader.cell_prop_keys, start=1):
        vec = graph.x[:, irow]

        vec_ew = vec * e_weight
        mean = torch.mean(vec_ew)
        std = torch.std(vec_ew)
        hlvs[key + "_mean_ew"] = mean
        hlvs[key + "_std_ew"] = std
        # hlvs[key + "_mom3_ew"] = self._stand_mom(vec, mean, std, 3)
        # hlvs[key + "_mom4_ew"] = self._stand_mom(vec, mean, std, 4)
        for var, v in hlvs.items():
            if not var.startswith(key):
                continue
            if torch.isnan(v):
                logger.error(f"NaN Computed for hlv {var}")
    return hlvs


def stand_mom(
    vec: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, order: int
) -> torch.Tensor:
    numerator = torch.mean(torch.pow(vec - mean, order))
    denominator = torch.pow(std, order / 2.0)
    if numerator == denominator:
        return torch.tensor(1).float()
    if denominator == 0:
        raise ValueError("Would devide by 0.")
    return numerator / denominator


def min_mean_max(vec):
    return (
        f"min {torch.min(vec)} mean {torch.mean(vec)} max"
        f" {torch.max(vec)} nan%{sum(torch.isnan(vec))/sum(vec.shape)}"
    )


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