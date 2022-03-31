from typing import List

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool


# construct the branching for each graph individually
def reverse_construct_tree(graph: Data, branches: List[int]) -> Data:
    # set the last level
    graph.levels: List[torch.Tensor] = [graph.x]
    graph.children: List[torch.Tensor] = []

    # reverse construct the tree
    cur_nodes = graph.x.shape[0]
    for n_branches in branches[::-1]:
        # randomly assign each node
        assert cur_nodes % n_branches == 0
        # create the vector that clusters the nodes
        branching = torch.tensor(
            np.arange(cur_nodes // n_branches).repeat(n_branches),
            dtype=torch.long,
        )
        branching = branching[torch.randperm(branching.size()[0])]

        # np.random.shuffle(branching)
        cur_nodes = cur_nodes // n_branches
        graph.levels.append(global_mean_pool(graph.levels[-1], branching))
        # reverse the mapping of the of the branchings
        graph.children.append(torch.argsort(branching).reshape(-1, n_branches))

    graph.levels.reverse()
    graph.children.reverse()

    # for ilevel, level in enumerate(graph.levels):
    #     setattr(
    #         graph,
    #         f"level_{ilevel}",
    #         level,
    #     )
    # for ichild, child in enumerate(graph.children):
    #     setattr(
    #         graph,
    #         f"branching_{ichild}",
    #         child,
    #     )

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

    # for ibranching, n_branches in list(enumerate(branches))[::-1]:

    #     current_batch_idxs = np.arange(batch_size).repeat(points)
    #     # Note: np.repeat repeats elementwise, not the whole array
    #     # The maximum of each event is the new number of points,
    #     # so we just multiply the batch_idxs with it
    #
    #     add_to_branching = torch.tensor(
    #         (current_batch_idxs * points),
    #         dtype=torch.long,
    #     )

    #     setattr(
    #         batch,
    #         f"branching_{ibranching}",
    #         add_to_branching + branching,
    #     )
    return batch
