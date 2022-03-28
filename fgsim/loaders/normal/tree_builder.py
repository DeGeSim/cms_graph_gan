import numpy as np
import torch
from torch_geometric.data import Batch, Data

from fgsim.config import conf, device


# construct the branching for each graph individually
def reverse_construct_tree(graph: Data) -> Data:
    cur_nodes = graph.x.shape[0]
    for ibranching, branches in list(enumerate(conf.tree.branches))[:0:-1]:
        # randomly assign each node
        assert cur_nodes % branches == 0
        branching = np.arange(cur_nodes // branches).repeat(branches)
        # np.random.shuffle(branching)

        setattr(
            graph,
            f"branching_{ibranching}",
            torch.tensor(branching, dtype=torch.long, device=device),
        )
        cur_nodes = cur_nodes // branches
    return graph


# The branching values created by the previous function have now been stacked
# to disentangle the events we need to add the values of batch.batch to the
# respective graph.branching_? vectors
def add_batch_to_branching(batch: Batch) -> Batch:
    points = conf.loader.max_points
    for ibranching, branches in list(enumerate(conf.tree.branches))[:0:-1]:
        branching = getattr(batch, f"branching_{ibranching}")
        # goal: create add_to_branching, so that each event has
        # separate indexes in the branching_? vectors
        current_batch_idxs = np.arange(conf.loader.batch_size).repeat(points)
        # Note: np.repeat repeats elementwise, not the whole array
        # The maximum of each event is the new number of points,
        # so we just multiply the batch_idxs with it
        points = points // branches
        add_to_branching = torch.tensor(
            (current_batch_idxs * points),
            dtype=torch.long,
            device=device,
        )

        setattr(
            batch,
            f"branching_{ibranching}",
            add_to_branching + branching,
        )
    return batch


def test_add_batch_to_branching(batch: Batch):
    from torch_scatter import scatter_mean

    res = batch.x
    for ibranching, branches in list(enumerate(conf.tree.branches))[:0:-1]:
        branching = getattr(batch, f"branching_{ibranching}")

        points = res.shape[0] // conf.loader.batch_size
        max_br_idx = (points // branches - 1) + (points // branches) * (
            conf.loader.batch_size - 1
        )
        assert max(branching) == max_br_idx
        res_new = scatter_mean(res, branching, dim=0)
        assert not torch.tensor([0, 0], dtype=torch.float) in res_new
        assert res.shape[0] / res_new.shape[0] == branches
        res = res_new

    assert res.shape[0] == conf.loader.batch_size
