from math import prod
from typing import List

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

from fgsim.models.branching.graph_tree import GraphTree


# construct the branching for each graph individually
def reverse_construct_tree(graph: Data, branches: List[int]) -> Data:
    # set the last level
    x_by_level: List[torch.Tensor] = [graph.x]
    children: List[torch.Tensor] = []

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
        x_by_level.append(global_mean_pool(x_by_level[-1], branching))
        # reverse the mapping of the of the branchings
        children.append(torch.argsort(branching).reshape(-1, n_branches))

    x_by_level.reverse()
    children.reverse()
    idxs_by_level = [
        torch.arange(len(level), dtype=torch.long)
        + sum([len(lvl) for lvl in x_by_level[:ilevel]])
        for ilevel, level in enumerate(x_by_level)
    ]
    graph.x = torch.vstack(x_by_level)
    graph.children = children
    graph.idxs_by_level = idxs_by_level
    # graph.batch=torch.arange(sum([len(lvl) for lvl in x_by_level]), dtype=torch.long),

    # for ilevel, level in enumerate(levels):
    #     setattr(
    #         graph,
    #         f"level_{ilevel}",
    #         level,
    #     )
    # for ichild, child in enumerate(children):
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
) -> GraphTree:
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

    return GraphTree(
        x=batch.x,
        batch=batch.batch,
        children=batch.children,
        idxs_by_level=batch.idxs_by_level,
    )

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
    # return batch
