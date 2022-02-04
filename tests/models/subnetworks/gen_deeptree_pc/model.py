import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv


def test_GlobalFeedBackNN_ancestor_conv(static_objects):
    graph = static_objects.graph
    branching_layer = static_objects.branching_layer
    dyn_hlvs_layer = static_objects.dyn_hlvs_layer
    ancestor_conv_layer = static_objects.ancestor_conv_layer
    n_global = static_objects.props["n_global"]
    n_levels = static_objects.props["n_levels"]
    for _ in range(n_levels):
        # ### Global
        global_features = dyn_hlvs_layer(graph)
        assert global_features.shape[1] == n_global
        graph = branching_layer(graph, global_features)
        graph.x = ancestor_conv_layer(
            graph.edge_index,
            torch.hstack([graph.x, global_features[graph.event]]),
        )


def test_GlobalFeedBackNN_GINConv(static_objects):
    graph = static_objects.graph
    branching_layer = static_objects.branching_layer
    dyn_hlvs_layer = static_objects.dyn_hlvs_layer
    n_global = static_objects.props["n_global"]
    n_levels = static_objects.props["n_levels"]
    n_features = static_objects.props["n_features"]
    conv = GINConv(
        nn.Sequential(
            nn.Linear(n_features + n_global, n_features),
        )
    )

    for _ in range(n_levels):
        # ### Global
        global_features = dyn_hlvs_layer(graph)
        assert global_features.shape[1] == n_global
        graph = branching_layer(graph, global_features)
        graph.x = conv(
            x=torch.hstack([graph.x, global_features[graph.event]]),
            edge_index=graph.edge_index,
        )


def test_full_NN_compute_graph(static_objects):
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
    dyn_hlvs_layer = static_objects.dyn_hlvs_layer
    ancestor_conv_layer = static_objects.ancestor_conv_layer
    n_global = static_objects.props["n_global"]
    n_levels = static_objects.props["n_levels"]

    zero_feature = torch.zeros_like(graph.x[0])
    x_old = graph.x
    for ilevel in range(n_levels):
        if ilevel > 0:
            leaf = branching_layer.tree[ilevel][0]
            pc_leaf_point = graph.x[leaf.idxs[2]]
            sum(pc_leaf_point).backward(retain_graph=True)

            assert x_old.grad is not None
            assert torch.all(x_old.grad[0] == zero_feature)
            assert torch.all(x_old.grad[1] == zero_feature)
            assert torch.any(x_old.grad[2] != zero_feature)

        global_features = dyn_hlvs_layer(graph)
        assert global_features.shape[1] == n_global
        graph = branching_layer(graph, global_features)
        graph.x = ancestor_conv_layer(graph, global_features)
