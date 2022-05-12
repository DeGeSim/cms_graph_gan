from math import prod

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from fgsim.loaders.normal.seq import cluster_graph_random
from fgsim.loaders.normal.tree_builder import (
    add_batch_to_branching,
    reverse_construct_tree,
)
from fgsim.models.branching.graph_tree import GraphTreeWrapper


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
    graph = gen_graph(points)

    branchings_list = cluster_graph_random(graph, branches)
    graph_tree = GraphTreeWrapper(
        reverse_construct_tree(
            graph, branchings_list=branchings_list, branches=branches
        )
    )

    for ilevel in range(len(branches)):
        mean_of_children = graph_tree.tftx_by_level[ilevel + 1][
            graph_tree.children[ilevel]
        ].mean(dim=1)
        assert torch.allclose(mean_of_children, graph_tree.tftx_by_level[ilevel])


def test_add_batch_to_branching():
    branches = [1, 3]
    batch_size = 3
    points = prod(branches)

    event_list = []
    for _ in range(batch_size):
        batch = gen_graph(points)
        event_list.append(
            reverse_construct_tree(
                batch,
                branches,
                branchings_list=cluster_graph_random(batch, branches),
            )
        )
    batch = Batch.from_data_list(event_list)
    graph_tree = GraphTreeWrapper(
        add_batch_to_branching(batch, branches, batch_size)
    )

    assert list(graph_tree.idxs_by_level[0]) == [0, 5, 10]
    assert list(graph_tree.idxs_by_level[1]) == [1, 6, 11]
    assert list(graph_tree.idxs_by_level[2]) == [2, 3, 4, 7, 8, 9, 12, 13, 14]
    assert list(batch.tbatch) == [0] * 5 + [1] * 5 + [2] * 5
    assert torch.all(
        torch.hstack(graph_tree.idxs_by_level).sort().values
        == torch.arange(len(graph_tree.tftx))
    )

    for ilevel in range(len(branches)):
        mean_of_children = graph_tree.tftx_by_level[ilevel + 1][
            graph_tree.children[ilevel]
        ].mean(dim=1)
        assert torch.allclose(mean_of_children, graph_tree.tftx_by_level[ilevel])

    for ilevel in range(len(branches) + 1):
        assert torch.all(
            graph_tree.batch_by_level[ilevel]
            == graph_tree.tbatch[graph_tree.idxs_by_level[ilevel]]
        )

    event_list = [GraphTreeWrapper(e) for e in event_list]
    for ievent in range(batch_size):
        assert torch.all(
            event_list[ievent].tftx
            == graph_tree.tftx[(graph_tree.tbatch == ievent)]
        )

    for ilevel in range(len(branches) + 1):
        level = torch.vstack([e.tftx_by_level[ilevel] for e in event_list])
        assert torch.all(level == graph_tree.tftx[graph_tree.idxs_by_level[ilevel]])
        assert torch.all(level == graph_tree.tftx_by_level[ilevel])

    for ilevel in range(len(branches)):
        for ievent in range(batch_size):
            levelarr = torch.zeros(len(graph_tree.tftx), dtype=torch.long)
            levelarr[graph_tree.idxs_by_level[ilevel]] = 1
            levelarr = levelarr > 0
            assert torch.all(
                event_list[ievent].tftx_by_level[ilevel]
                == graph_tree.tftx[(graph_tree.tbatch == ievent) & levelarr]
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
