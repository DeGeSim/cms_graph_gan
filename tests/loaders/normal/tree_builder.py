from math import prod

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from fgsim.loaders.normal.tree_builder import (
    add_batch_to_branching,
    reverse_construct_tree,
)


def gen_graph(points):
    mu = [100, 100]
    covar = [[100.0, 50], [50, 100]]
    x1 = np.random.multivariate_normal(mu, covar, points)
    pointcloud = torch.tensor(x1).float()
    graph = Data(x=pointcloud)
    return graph


def test_reverse_construct_tree():
    branches = [2, 5]
    points = prod(branches)
    graph = reverse_construct_tree(gen_graph(points), branches=branches)

    for ilevel in range(len(branches)):
        mean_of_children = graph.x_by_level[ilevel + 1][
            graph.children[ilevel]
        ].mean(dim=1)
        assert torch.allclose(mean_of_children, graph.x_by_level[ilevel])


def test_add_batch_to_branching():
    branches = [1, 3]
    batch_size = 3
    points = prod(branches)

    event_list = [
        reverse_construct_tree(gen_graph(points), branches)
        for _ in range(batch_size)
    ]
    batch = Batch.from_data_list(event_list)
    batch = add_batch_to_branching(batch, branches, batch_size)

    assert torch.all(
        torch.hstack(batch.idxs_by_level).sort().values
        == torch.arange(len(batch.x))
    )

    for ilevel in range(len(branches)):
        mean_of_children = batch.x_by_level[ilevel + 1][
            batch.children[ilevel]
        ].mean(dim=1)
        assert torch.allclose(mean_of_children, batch.x_by_level[ilevel])

    for ilevel in range(len(branches) + 1):
        assert torch.all(
            batch.batch_by_level[ilevel] == batch.batch[batch.idxs_by_level[ilevel]]
        )
    for ievent in range(batch_size):
        assert torch.all(event_list[ievent].x == batch.x[(batch.batch == ievent)])

    for ilevel in range(len(branches) + 1):
        level = torch.vstack([e.x_by_level[ilevel] for e in event_list])
        assert torch.all(level == batch.x[batch.idxs_by_level[ilevel]])
        assert torch.all(level == batch.x_by_level[ilevel])

    for ilevel in range(len(branches)):
        for ievent in range(batch_size):
            levelarr = torch.zeros(len(batch.x), dtype=torch.long)
            levelarr[batch.idxs_by_level[ilevel]] = 1
            levelarr = levelarr > 0
            assert torch.all(
                event_list[ievent].x_by_level[ilevel]
                == batch.x[(batch.batch == ievent) & levelarr]
            )
    # from torch_scatter import scatter_mean

    # res = batch.x
    # for ibranching, branches in list(enumerate(branches))[::-1]:
    #     branching = getattr(batch, f"branching_{ibranching}")

    #     points = res.shape[0] // batch_size
    #     max_br_idx = (points // branches - 1) + (points // branches) * (
    #         batch_size - 1
    #     )
    #     assert max(branching) == max_br_idx
    #     res_new = scatter_mean(res, branching, dim=0)
    #     assert not torch.tensor([0, 0], dtype=torch.float) in res_new
    #     assert res.shape[0] / res_new.shape[0] == branches
    #     res = res_new

    # assert res.shape[0] == batch_size
