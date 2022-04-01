from typing import List

import numpy as np
import pytest
import torch
from conftest import DTColl

from fgsim.models.branching.branching import reshape_features
from fgsim.models.branching.node import Node

device = torch.device("cpu")


def test_BranchingLayer_compute_graph(static_objects: DTColl):
    """
    Make sure that the events are independent.
    For this, we apply branching and make sure, that the gradient only
    is nonzero for the root of the event we apply the `backwards()` on.
    Args:
      graph (Data): The original graph.
      branching_layers[0] (BranchingLayer): The branching layer to test.
      global_features (torch.Tensor): torch.Tensor
    """
    graph = static_objects.graph
    branching_layers = static_objects.branching_layers

    new_graph1 = branching_layers[0](graph)
    tree = branching_layers[0].tree
    assert torch.all(
        new_graph1.x[new_graph1.level_idxs[1]]
        == new_graph1.x[torch.hstack([e.idxs for e in tree.tree_lists[1]])]
    )
    leaf = tree.tree_lists[1][0]

    pc_leaf_point = new_graph1.x[leaf.idxs[2]]
    sum(pc_leaf_point).backward(retain_graph=True)

    zero_feature = torch.zeros_like(graph.x[0])
    assert graph.x.grad is not None
    assert torch.all(graph.x.grad[0] == zero_feature)
    assert torch.all(graph.x.grad[1] == zero_feature)
    assert torch.any(graph.x.grad[2] != zero_feature)

    new_graph2 = branching_layers[1](new_graph1)
    assert torch.all(
        new_graph2.x[new_graph2.level_idxs[2]]
        == new_graph2.x[torch.hstack([e.idxs for e in tree.tree_lists[2]])]
    )

    leaf = branching_layers[1].tree.tree_lists[2][0]
    pc_leaf_point = new_graph2.x[leaf.idxs[2]]
    sum(pc_leaf_point).backward()

    assert graph.x.grad is not None
    assert torch.all(graph.x.grad[0] == zero_feature)
    assert torch.all(graph.x.grad[1] == zero_feature)
    assert torch.any(graph.x.grad[2] != zero_feature)


def test_BranchingLayer_connectivity_static(static_objects: DTColl):
    props = static_objects.props
    graph = static_objects.graph
    branching_layers = static_objects.branching_layers
    for ilevel in range(1, props["n_levels"]):
        graph = branching_layers[ilevel - 1](graph)
        conlist = graph.edge_index.T.cpu().numpy().tolist()
        connections = {tuple(x) for x in conlist}
        # Static Check
        if ilevel == 1:
            # F|B
            # -|-
            # 0|0
            # 1|0
            # 2|0
            # 3|0
            # 4|1
            # 5|0
            # 6|1
            # 7|0
            # 8|1
            # check the connections
            expected_connections = {(0, 3), (1, 4), (0, 6), (1, 7), (2, 5), (2, 8)}
            assert connections.issuperset(expected_connections)
        if ilevel == 2:
            expected_connections = {
                (3, 9),
                (3, 12),
                (4, 10),
                (4, 13),
                (5, 11),
                (5, 14),
                (6, 15),
                (6, 18),
                (7, 16),
                (7, 19),
                (8, 17),
                (8, 20),
            }
            assert connections.issuperset(expected_connections)
        # No double connections
        assert len(connections) == len(conlist)


def test_BranchingLayer_connectivity_dyn(dyn_objects: DTColl):
    props = dyn_objects.props
    graph = dyn_objects.graph
    branching_layers = dyn_objects.branching_layers
    n_features = props["n_features"]
    n_branches = props["n_branches"]
    # n_global = props["n_global"]
    batch_size = props["batch_size"]
    n_levels = props["n_levels"]
    # Shape
    tree_lists = branching_layers[0].tree.tree_lists
    assert len(tree_lists) == n_levels
    for ilevel in range(1, n_levels):
        n_parents = len(tree_lists[ilevel - 1])
        assert len(tree_lists[ilevel]) == n_parents * n_branches

    for ilevel in range(n_levels):
        if ilevel > 0:
            graph = branching_layers[ilevel - 1](graph)
        # split once
        # x shape testing
        assert graph.x.shape[1] == n_features
        assert graph.x.shape[0] == batch_size * sum(
            [n_branches ** i for i in range(ilevel + 1)]
        )
        # edge_index shape testing
        assert graph.edge_index.shape[0] == 2
        # Number of connections
        # Sum n_branches^ilayer*ilayer for ilayer in 0..nlayers
        assert graph.edge_index.shape[1] == batch_size * sum(
            [n_branches ** i * i for i in range(ilevel + 1)]
        )

        conlist = graph.edge_index.T.cpu().numpy().tolist()
        connections = {tuple(x) for x in conlist}
        # No double connections
        assert len(connections) == len(conlist)

    conlist = graph.edge_index.T.cpu().numpy().tolist()
    connections = {tuple(x) for x in conlist}

    def recurr_check_connection(node: Node, ancestors_idxs: List[np.ndarray]):
        new_ancestors_idxs = [node.idxs.cpu().numpy()] + ancestors_idxs
        for idxs in new_ancestors_idxs:
            for child in node.children:
                child_idxs = child.idxs.cpu().numpy()
                for source, target in zip(
                    idxs,
                    child_idxs,
                ):
                    assert (source, target) in connections
                recurr_check_connection(child, new_ancestors_idxs)

    recurr_check_connection(tree_lists[0][0], [])


# Test the reshaping
def demo_mtx(*, n_parents, batch_size, n_branches, n_features):
    mtx = torch.ones(n_parents * batch_size, n_branches * n_features)
    i, j = 0, 0
    ifield = 0
    for iparent in range(n_parents):
        for ibranch in range(n_branches):
            for ievent in range(batch_size):
                for ifeature in range(n_features):
                    # print(f"Acessing {i}, {j+ifeature}")
                    mtx[i, ifeature + ibranch * n_features] = ifield
                ifield = ifield + 1
                i = i + 1
            i = i - batch_size
            j = j + 1
        i = i + batch_size
        j = j - n_branches
    return mtx


@pytest.mark.parametrize("n_parents", [4])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("n_branches", [3])
@pytest.mark.parametrize("n_features", [5])
def test_reshape_features(n_parents, batch_size, n_branches, n_features):
    mtx = demo_mtx(
        n_parents=n_parents,
        batch_size=batch_size,
        n_branches=n_branches,
        n_features=n_features,
    )
    mtx_reshaped = reshape_features(
        mtx,
        n_parents=n_parents,
        batch_size=batch_size,
        n_branches=n_branches,
        n_features=n_features,
    )
    for i in range(n_parents * batch_size):
        assert torch.all(mtx_reshaped[i, :] == i)
    torch.all(
        torch.arange(n_parents * batch_size * n_branches) == mtx_reshaped[:, 0]
    )


# def test_branching_by_training():
#     from torch_geometric.data import Batch, Data

#     from fgsim.models.branching.branching import BranchingLayer, Tree
#     from fgsim.models.dnn_gen import dnn_gen

#     n_features = 2
#     batch_size = 1
#     n_branches = 2
#     n_levels = 3
#     n_global = 0
#     device = torch.device("cpu")
#     tree = Tree(
#         batch_size=batch_size,
#         n_features=n_features,
#         n_branches=n_branches,
#         n_levels=n_levels,
#         device=device,
#     )
#     branching_layer = BranchingLayer(
#         tree=tree,
#         proj_nn=dnn_gen(
#             n_features + n_global,
#             n_features * n_branches,
#             n_layers=4,
#         ).to(device),
#     )
#     batch = Batch.from_data_list([Data(x=torch.tensor([[1.0, 1.0]]))])
#     global_features = torch.tensor([[]])
#     target = torch.tensor([[4.0, 7.0], [5.0, 1.0], [2.0, 2.5], [3.0, 3.5]])
#     loss_fn = torch.nn.MSELoss()
#     optimizer = torch.optim.Adam(branching_layer.parameters())
#     # check the branching layer
#     for _ in range(10000):
#         optimizer.zero_grad()
#         b1 = branching_layer(batch)
#         b2 = branching_layer(b1)
#         loss = loss_fn(b2.x[3:, :], target)
#         loss.backward()
#         optimizer.step()
#         if _ % 1000 == 0:
#             print(loss)
#         if torch.allclose(target, b2.x[3:, :], rtol=1e-4, atol=1e-4):
#             return
#     raise Exception
