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
            x=graph.tftx,
            edge_index=tree.ancestor_ei(ilevel),
            edge_attr=tree.ancestor_ea(ilevel),
            batch=graph.tbatch,
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
                x=graph.tftx,
                edge_index=tree.ancestor_ei(ilevel),
                edge_attr=tree.ancestor_ea(ilevel),
                batch=graph.tbatch,
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


def test_full_modelparts_grad():
    """
    Make sure that the events are independent.
    For this, we apply branching and make sure, that the gradient only
    is nonzero for the root of the event we apply the `backwards()` on.
    Args:
      graph (Data): The original graph.
      branching_layer (BranchingLayer): The branching layer to test.
      global_features (torch.Tensor): torch.Tensor
    """
    from fgsim.config import conf, conf_without_paths, device
    from fgsim.models.gen.gen_deeptree import (
        GraphTreeWrapper,
        ModelClass,
        TreeGenType,
    )

    # normalization needs to be set to false, otherwise Batchnorm
    # will propagate some gradient betweeen the events
    conf.ffn.normalize = False
    model = ModelClass(**conf_without_paths.model_param_options.gen_deeptree).to(
        device
    )

    z = torch.randn(*model.z_shape, requires_grad=True, device=device)

    def check_z():
        assert torch.all(z.grad[torch.arange(len(z)) != 1] == 0), "Tainted gradient"
        assert torch.any(z.grad[1] != 0), "Grad not propagated"
        z.grad = None

    batch_size = model.batch_size
    features = model.features
    branches = model.branches
    n_levels = len(branches)

    # Init the graph object
    graph_tree = GraphTreeWrapper(
        TreeGenType(
            tftx=z.reshape(batch_size, features[0]),
            batch_size=batch_size,
        )
    )
    # check
    graph_tree.tftx[[graph_tree.tbatch == 1]].sum().backward(retain_graph=True)
    check_z()

    # Do the branching
    for ilevel in range(n_levels):
        # Assign the global features
        graph_tree.global_features = model.dyn_hlvs_layers[ilevel](
            graph_tree.tftx_by_level[ilevel][..., : features[-1]],
            graph_tree.tbatch[graph_tree.idxs_by_level[ilevel]],
        )
        graph_tree.global_features[1].sum().backward(retain_graph=True)
        check_z()
        graph_tree = model.branching_layers[ilevel](graph_tree)

        graph_tree.tftx[[graph_tree.tbatch == 1]].sum().backward(retain_graph=True)
        check_z()

        graph_tree.tftx = model.ancestor_conv_layers[ilevel](
            x=graph_tree.tftx,
            edge_index=model.tree.ancestor_ei(ilevel),
            edge_attr=model.tree.ancestor_ea(ilevel),
            batch=graph_tree.tbatch,
            global_features=graph_tree.global_features,
        )
        graph_tree.tftx[[graph_tree.tbatch == 1]].sum().backward(retain_graph=True)
        check_z()

        graph_tree.tftx = model.child_conv_layers[ilevel](
            x=graph_tree.tftx,
            edge_index=model.tree.children_ei(ilevel),
            batch=graph_tree.tbatch,
            global_features=graph_tree.global_features,
        )
        graph_tree.tftx[[graph_tree.tbatch == 1]].sum().backward(retain_graph=True)
        check_z()

    graph_tree.data.x = graph_tree.tftx_by_level[-1]
    graph_tree.data.batch = graph_tree.batch_by_level[-1]

    graph_tree.data.x[graph_tree.data.batch == 1].sum().backward(retain_graph=True)

    check_z()


def test_full_model_grad():
    """
    Make sure that the events are independent.
    For this, we apply branching and make sure, that the gradient only
    is nonzero for the root of the event we apply the `backwards()` on.
    Args:
      graph (Data): The original graph.
      branching_layer (BranchingLayer): The branching layer to test.
      global_features (torch.Tensor): torch.Tensor
    """
    from fgsim.config import conf, conf_without_paths, device
    from fgsim.models.gen.gen_deeptree import ModelClass

    conf.ffn.normalize = False

    model = ModelClass(**conf_without_paths.model_param_options.gen_deeptree).to(
        device
    )

    z = torch.randn(*model.z_shape, requires_grad=True, device=device)
    batch = model(z)
    sum_of_features = batch.x[batch.batch == 1].sum()
    sum_of_features.backward(retain_graph=True)
    zero_feature = torch.zeros_like(z[0])
    assert torch.all(z.grad[0] == zero_feature)
    assert torch.any(z.grad[1] != zero_feature)
    assert torch.all(z.grad[2] == zero_feature)
