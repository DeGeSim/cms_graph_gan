from typing import List

import numpy as np
import pytest
import torch

from fgsim.models.branching.branching import reshape_features
from fgsim.models.branching.node import Node

device = torch.device("cpu")


def test_BranchingLayer_compute_graph(static_objects):
    """
    Make sure that the events are independent.
    For this, we apply branching and make sure, that the gradient only
    is nonzero for the root of the event we apply the `backwards()` on.
    Args:
      graph (Data): The original graph.
      branching_layer (BranchingLayer): The branching layer to test.
      global_features (torch.Tensor): torch.Tensor
    """
    graph = static_objects.graph
    branching_layer = static_objects.branching_layer
    global_features = static_objects.global_features

    new_graph1 = branching_layer(graph, global_features)
    leaf = branching_layer.tree.tree_lists[1][0]
    pc_leaf_point = new_graph1.x[leaf.idxs[2]]
    sum(pc_leaf_point).backward(retain_graph=True)

    zero_feature = torch.zeros_like(graph.x[0])
    assert graph.x.grad is not None
    assert torch.all(graph.x.grad[0] == zero_feature)
    assert torch.all(graph.x.grad[1] == zero_feature)
    assert torch.any(graph.x.grad[2] != zero_feature)

    new_graph2 = branching_layer(new_graph1, global_features)
    leaf = branching_layer.tree.tree_lists[2][0]
    pc_leaf_point = new_graph2.x[leaf.idxs[2]]
    sum(pc_leaf_point).backward()

    assert graph.x.grad is not None
    assert torch.all(graph.x.grad[0] == zero_feature)
    assert torch.all(graph.x.grad[1] == zero_feature)
    assert torch.any(graph.x.grad[2] != zero_feature)


def test_BranchingLayer_connectivity_static(static_objects):
    props = static_objects.props
    graph = static_objects.graph
    branching_layer = static_objects.branching_layer
    global_features = static_objects.global_features
    for ilevel in range(props["n_levels"] - 1):
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

        graph = branching_layer(graph, global_features)


def test_BranchingLayer_connectivity_dyn(dyn_objects):
    props = dyn_objects.props
    graph = dyn_objects.graph
    branching_layer = dyn_objects.branching_layer
    global_features = dyn_objects.global_features
    n_features = props["n_features"]
    n_branches = props["n_branches"]
    # n_global = props["n_global"]
    n_events = props["n_events"]
    n_levels = props["n_levels"]
    # Shape
    for ilevel in range(1, n_levels):
        n_parents = len(branching_layer.tree.tree_lists[ilevel - 1])
        assert (
            len(branching_layer.tree.tree_lists[ilevel]) == n_parents * n_branches
        )

    for ilevel in range(n_levels):
        # split once
        # x shape testing
        assert graph.x.shape[1] == n_features
        assert graph.x.shape[0] == n_events * sum(
            [n_branches ** i for i in range(ilevel + 1)]
        )
        # edge_index shape testing
        assert graph.edge_index.shape[0] == 2
        # Number of connections
        # Sum n_branches^ilayer*ilayer for ilayer in 0..nlayers
        assert graph.edge_index.shape[1] == n_events * sum(
            [n_branches ** i * i for i in range(ilevel + 1)]
        )

        conlist = graph.edge_index.T.cpu().numpy().tolist()
        connections = {tuple(x) for x in conlist}
        # No double connections
        assert len(connections) == len(conlist)

        graph = branching_layer(graph, global_features)

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

    recurr_check_connection(branching_layer.tree.tree_lists[0][0], [])


# Test the reshaping
def demo_mtx(*, n_parents, n_events, n_branches, n_features):
    mtx = torch.ones(n_parents * n_events, n_branches * n_features)
    i, j = 0, 0
    ifield = 0
    for iparent in range(n_parents):
        for ibranch in range(n_branches):
            for ievent in range(n_events):
                for ifeature in range(n_features):
                    # print(f"Acessing {i}, {j+ifeature}")
                    mtx[i, ifeature + ibranch * n_features] = ifield
                ifield = ifield + 1
                i = i + 1
            i = i - n_events
            j = j + 1
        i = i + n_events
        j = j - n_branches
    return mtx


@pytest.mark.parametrize("n_parents", [4])
@pytest.mark.parametrize("n_events", [2])
@pytest.mark.parametrize("n_branches", [3])
@pytest.mark.parametrize("n_features", [5])
def test_reshape_features(n_parents, n_events, n_branches, n_features):
    mtx = demo_mtx(
        n_parents=n_parents,
        n_events=n_events,
        n_branches=n_branches,
        n_features=n_features,
    )
    mtx_reshaped = reshape_features(
        mtx,
        n_parents=n_parents,
        n_events=n_events,
        n_branches=n_branches,
        n_features=n_features,
    )
    for i in range(n_parents * n_events):
        assert torch.all(mtx_reshaped[i, :] == i)
    torch.all(torch.arange(n_parents * n_events * n_branches) == mtx_reshaped[:, 0])
