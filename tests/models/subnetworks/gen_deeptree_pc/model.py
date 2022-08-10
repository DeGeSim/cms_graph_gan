import torch
import torch.nn as nn
from conftest import DTColl
from torch_geometric.nn.conv import GINConv


def test_GlobalFeedBackNN_ancestor_conv(static_objects: DTColl):
    graph = static_objects.graph
    branching_layers = static_objects.branching_layers
    dyn_hlvs_layer = static_objects.dyn_hlvs_layer
    ancestor_conv_layer = static_objects.ancestor_conv_layer
    tree = static_objects.tree
    n_global = static_objects.props["n_global"]
    n_levels = static_objects.props["n_levels"]
    for ilevel in range(n_levels):
        if ilevel > 0:
            graph = branching_layers[ilevel - 1](graph)
        # ### Global
        graph.global_features = dyn_hlvs_layer(graph.tftx, graph.tbatch)
        assert graph.global_features.shape[1] == n_global
        graph.tftx = ancestor_conv_layer(
            tftx=graph.tftx,
            edge_index=tree.ancestor_ei(ilevel),
            edge_attr=tree.ancestor_ea(ilevel),
            tbatch=graph.tbatch,
            global_features=graph.global_features,
        )


def test_GlobalFeedBackNN_GINConv(static_objects: DTColl):
    graph = static_objects.graph
    branching_layers = static_objects.branching_layers
    dyn_hlvs_layer = static_objects.dyn_hlvs_layer
    tree = static_objects.tree
    n_global = static_objects.props["n_global"]
    n_levels = static_objects.props["n_levels"]
    n_features = static_objects.props["n_features"]
    conv = GINConv(
        nn.Sequential(
            nn.Linear(n_features + n_global, n_features),
        )
    )

    for ilevel in range(n_levels):
        if ilevel > 0:
            graph = branching_layers[ilevel - 1](graph)
        # ### Global
        global_features = dyn_hlvs_layer(graph.tftx, graph.tbatch)
        assert global_features.shape[1] == n_global

        graph.tftx = conv(
            x=torch.hstack([graph.tftx, global_features[graph.tbatch]]),
            edge_index=tree.ancestor_ei(ilevel),
        )


def test_full_NN_compute_graph(static_objects: DTColl):
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
    branching_layers = static_objects.branching_layers
    dyn_hlvs_layer = static_objects.dyn_hlvs_layer
    ancestor_conv_layer = static_objects.ancestor_conv_layer
    tree = static_objects.tree
    n_global = static_objects.props["n_global"]
    n_levels = static_objects.props["n_levels"]

    tree_lists = branching_layers[0].tree.tree_lists
    zero_feature = torch.zeros_like(graph.tftx[0])
    x_old = graph.tftx
    for ilevel in range(n_levels):
        if ilevel > 0:
            graph = branching_layers[ilevel - 1](graph)
            graph.tftx = ancestor_conv_layer(
                tftx=graph.tftx,
                edge_index=tree.ancestor_ei(ilevel),
                edge_attr=tree.ancestor_ea(ilevel),
                tbatch=graph.tbatch,
            )
            leaf = tree_lists[ilevel][0]
            pc_leaf_point = graph.tftx[leaf.idxs[2]]
            sum(pc_leaf_point).backward(retain_graph=True)

            assert x_old.grad is not None
            assert torch.all(x_old.grad[0] == zero_feature)
            assert torch.all(x_old.grad[1] == zero_feature)
            assert torch.any(x_old.grad[2] != zero_feature)

        global_features = dyn_hlvs_layer(graph.tftx, graph.tbatch)
        assert global_features.shape[1] == n_global
