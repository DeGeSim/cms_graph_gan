from math import prod
from typing import List

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool


# construct the branching for each graph individually
def reverse_construct_tree(
    graph: Data,
    branches: List[int],
    branchings_list: List[torch.Tensor],
) -> Data:
    # set the last level
    tftx_by_level: List[torch.Tensor] = [graph.x]
    children: List[torch.Tensor] = []

    # reverse construct the tree
    cur_nodes = graph.x.shape[0]
    for ilevel, n_branches in reversed(list(enumerate(branches))):
        # randomly assign each node
        assert cur_nodes % n_branches == 0
        # create the vector that clusters the nodes

        branching_test_example = torch.tensor(
            np.arange(cur_nodes // n_branches).repeat(n_branches),
            dtype=torch.long,
        )
        # branching = branching_test_example[
        #     torch.randperm(branching_test_example.size()[0])
        # ]
        branching = branchings_list[ilevel]
        assert len(branching) == (cur_nodes // n_branches) * n_branches
        assert torch.all(branching_test_example == torch.sort(branching)[0])

        # np.random.shuffle(branching)
        cur_nodes = cur_nodes // n_branches
        tftx_by_level.append(global_mean_pool(tftx_by_level[-1], branching))
        # reverse the mapping of the of the branchings
        children.append(torch.argsort(branching).reshape(-1, n_branches))

    tftx_by_level.reverse()
    children.reverse()
    idxs_by_level = [
        torch.arange(len(level), dtype=torch.long)
        + sum([len(lvl) for lvl in tftx_by_level[:ilevel]])
        for ilevel, level in enumerate(tftx_by_level)
    ]
    graph.tftx = torch.vstack(tftx_by_level)
    graph.children = children
    graph.idxs_by_level = idxs_by_level

    return graph


# The branching values created by the previous function have now been stacked
# to disentangle the events we need to add the values of batch.batch to the
# respective graph.branching_? vectors
def add_batch_to_branching(
    batch: Batch, branches: List[int], batch_size: int
) -> Batch:
    points = 1
    for ilevel, n_branches in enumerate(branches):
        # goal: create add_to_branching, so that each event has
        # separate indexes in the branching_? vectors
        add_to_branching = (
            (
                np.arange(batch_size)
                .repeat(n_branches * points)
                .reshape(-1, n_branches)
            )
            * n_branches
            * points
        )

        batch.children[ilevel] += add_to_branching
        points *= n_branches

    sum_points = sum(
        [prod(branches[:ilevel]) for ilevel in range(len(branches) + 1)]
    )

    for ilevel in range(len(branches) + 1):
        points = prod(branches[:ilevel])
        add_to_idxs_by_level = np.arange(batch_size).repeat(points) * sum_points
        batch.idxs_by_level[ilevel] += add_to_idxs_by_level

    batch.tbatch = torch.arange(batch_size).repeat_interleave(sum_points)

    return batch
