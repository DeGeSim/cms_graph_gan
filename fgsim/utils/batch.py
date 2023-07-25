from collections import defaultdict

import torch
import torch_scatter
from torch_geometric.data import Batch, Data


def test():
    batch_size = 3
    node_range = (1, 10)
    nodes_v = torch.randint(*node_range, (batch_size,))
    x_list = [torch.rand(n, 3) for n in nodes_v]

    batch_list = [Data(x=x) for x in x_list]
    batch_truth = Batch.from_data_list(batch_list)

    batch_idx = batch_truth.batch

    batch = init_batch(batch_idx)

    add_nodewise_attr(batch, "x", torch.vstack(x_list))

    compare(batch, batch_truth)


def init_batch(batch_idx: torch.Tensor):
    if not batch_idx.dtype == torch.long:
        raise Exception("Batch index dtype must be torch.long")
    if not (batch_idx.diff() >= 0).all():
        raise Exception("Batch index must be increasing")
    if not batch_idx.dim() == 1:
        raise Exception()

    batch = Batch(batch=batch_idx)

    batch.ptr = ptr_from_batchidx(batch_idx)
    batch._num_graphs = int(batch.batch.max() + 1)

    batch._slice_dict = defaultdict(dict)
    batch._inc_dict = defaultdict(dict)
    return batch


def ptr_from_batchidx(batch_idx):
    # Construct the ptr to adress single graphs
    # graph[idx].x= batch.x[batch.ptr[idx]:batch.ptr[idx]+1]
    # Get delta with diff
    # Get idx of diff >0 with nonzero
    # shift by -1
    # add the batch size -1 as last element and add 0 in front
    dev = batch_idx.device
    return torch.concatenate(
        (
            torch.tensor(0).long().to(dev).unsqueeze(0),
            (batch_idx.diff()).nonzero().reshape(-1) + 1,
            torch.tensor(len(batch_idx)).long().to(dev).unsqueeze(0),
        )
    )


def add_nodewise_attr(batch: Batch, attrname: str, attr: torch.Tensor):
    device = attr.device
    assert device == batch.batch.device
    batch_idxs = batch.batch

    batch[attrname] = attr
    out = torch_scatter.scatter_add(
        torch.ones(len(attr), dtype=torch.long, device=device), batch_idxs, dim=0
    )
    out = out.cumsum(dim=0)
    batch._slice_dict[attrname] = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), out], dim=0
    )

    batch._inc_dict["x"] = torch.zeros(
        batch._num_graphs, dtype=torch.long, device=device
    )


def compare(ba: Batch, bb: Batch):
    if set(ba.keys) != set(bb.keys):
        raise Exception()
    for k in ba.keys:
        rec_comp(ba[k], bb[k])
        rec_comp(ba._slice_dict[k], bb._slice_dict[k])
        rec_comp(ba._inc_dict[k], bb._inc_dict[k])


def rec_comp(a, b):
    if not type(a) == type(b):
        raise Exception()
    if isinstance(a, dict):
        if not set(a.keys()) == set(b.keys()):
            raise Exception()
        for k in a:
            rec_comp(a[k], b[k])
    if isinstance(a, torch.Tensor):
        if not (a == b).all():
            raise Exception()


def fix_slice_dict_nodeattr(batch: Batch, attrname: str) -> Batch:
    if not hasattr(batch, "_slice_dict"):
        batch._slice_dict = defaultdict(dict)
    attr = batch[attrname]
    batch_idxs = batch.batch
    device = attr.device
    out = torch_scatter.scatter_add(
        torch.ones(len(attr), dtype=torch.long, device=device), batch_idxs, dim=0
    )
    out = out.cumsum(dim=0)
    batch._slice_dict[attrname] = torch.cat(
        [torch.zeros(1, dtype=torch.long, device=device), out], dim=0
    )
    return batch


test()
