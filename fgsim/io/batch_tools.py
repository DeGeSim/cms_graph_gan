from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch_scatter
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_add_pool, global_mean_pool

from fgsim.config import conf
from fgsim.monitoring.logger import logger


def pcs_to_batch_v1(pcs: torch.Tensor, events: torch.Tensor) -> Batch:
    pcs_list = [pcs[events == ievent] for ievent in range(max(events) + 1)]
    event_list = [Data(x=pc) for pc in pcs_list]

    return Batch.from_data_list(event_list)


def batch_sort_by_sort(
    pcs: torch.Tensor, batch_idxs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    reorder_idxs = batch_idxs.sort(stable=True)[1]
    pcs = pcs[reorder_idxs]
    batch_idxs = batch_idxs[reorder_idxs]
    return (pcs, batch_idxs)


def batch_sort_by_reshape(
    pcs: torch.Tensor, batch_idxs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_features = int(pcs.shape[1])
    n_events = int(batch_idxs.max() + 1)
    n_points_per_event = len(pcs) // n_events
    # events.reshape(-1,5).T
    assert torch.all(
        batch_idxs
        == torch.arange(n_events, device=pcs.device).repeat(n_points_per_event)
    )
    batch_idxs = batch_idxs.reshape(n_points_per_event, n_events).T.reshape(-1)
    pcs = (
        pcs.reshape(n_points_per_event, n_features * n_events)
        .T.reshape(n_events, n_features, n_points_per_event)
        .transpose(1, 2)
        .reshape(n_events, n_features * n_points_per_event)
    )
    return (pcs, batch_idxs)


def batch_construct_direct(pcs: torch.Tensor, batch_idxs: torch.Tensor) -> Batch:
    device = pcs.device
    batch = Batch(x=pcs, batch=batch_idxs)
    batch._num_graphs = int(batch.batch.max() + 1)

    _slice_dict = defaultdict(dict)
    out = torch_scatter.scatter_add(
        torch.ones(len(batch.x), dtype=torch.long, device=device), batch_idxs, dim=0
    )
    out = out.cumsum(dim=0)
    _slice_dict["x"] = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), out], dim=0
    )

    batch._slice_dict = _slice_dict

    _inc_dict = defaultdict(dict)
    _inc_dict["x"] = torch.zeros(batch._num_graphs, dtype=torch.long, device=device)
    batch._inc_dict = _inc_dict
    return batch


# 4 Options
def pcs_to_batch_sort_direct(pcs: torch.Tensor, batch_idxs: torch.Tensor) -> Batch:
    pcs, batch_idxs = batch_sort_by_sort(pcs, batch_idxs)
    batch = batch_construct_direct(pcs, batch_idxs)
    return batch


def pcs_to_batch_sort_list(pcs: torch.Tensor, batch_idxs: torch.Tensor) -> Batch:
    n_features = int(pcs.shape[1])
    n_events = int(batch_idxs.max() + 1)
    n_points_per_event = len(pcs) // n_events
    pcs, batch_idxs = batch_sort_by_sort(pcs, batch_idxs)
    pcs = pcs.reshape(n_events, n_points_per_event, n_features)
    batch = Batch.from_data_list([Data(x=e) for e in pcs])
    return batch


def pcs_to_batch_reshape_direct(
    pcs: torch.Tensor, batch_idxs: torch.Tensor
) -> Batch:
    n_features = int(pcs.shape[1])
    n_events = int(batch_idxs.max() + 1)
    n_points_per_event = len(pcs) // n_events
    pcs, batch_idxs = batch_sort_by_reshape(pcs, batch_idxs)
    batch = batch_construct_direct(
        pcs.reshape(n_events * n_points_per_event, n_features), batch_idxs
    )
    return batch


def pcs_to_batch_reshape_list(pcs: torch.Tensor, batch_idxs: torch.Tensor) -> Batch:
    n_features = int(pcs.shape[1])
    n_events = int(batch_idxs.max() + 1)
    n_points_per_event = len(pcs) // n_events
    pcs, batch_idxs = batch_sort_by_reshape(pcs, batch_idxs)
    pcs = pcs.reshape(n_events, n_points_per_event, n_features)
    batch = Batch.from_data_list([Data(x=e) for e in pcs])
    return batch


# fastest method
batch_from_pcs_list = pcs_to_batch_sort_list

# Compute stuff
def batch_compute_hlvs(batch: Batch) -> Batch:
    event_list = [x for x in batch.to_data_list()]
    for event in event_list:
        event.hlvs = compute_hlvs(event)
    batch = Batch.from_data_list(event_list)
    return batch


def compute_hlvs(graph: Data) -> Dict[str, torch.Tensor]:
    hlvs: Dict[str, torch.Tensor] = {}

    if "E" in conf.loader.cell_prop_keys:
        E_idx = conf.loader.cell_prop_keys.index("E")
        e_weight = graph.x[:, E_idx] / torch.sum(graph.x[:, E_idx])

    for irow, key in enumerate(conf.loader.cell_prop_keys):
        vec = graph.x[:, irow]
        hlvs[key + "_mean"] = torch.mean(vec)
        hlvs[key + "_std"] = torch.std(vec)
        if "E" in conf.loader.cell_prop_keys:
            if key == "E":
                continue
            vec_ew = vec * e_weight
            hlvs[key + "_mean_ew"] = torch.mean(vec_ew)
            hlvs[key + "_std_ew"] = torch.std(vec_ew)
    for var, v in hlvs.items():
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
