import torch
import torch.nn as nn
from conftest import DTColl
from torch_geometric.nn.conv import GINConv


def test_GlobalFeedBackNN_ancestor_conv(static_objects: DTColl):
    graph, tree, cond, branchings, dyn_hlvs_layer, ancestor_conv_layer = (
        static_objects.graph,
        static_objects.tree,
        static_objects.cond,
        static_objects.branchings,
        static_objects.dyn_hlvs_layer,
        static_objects.ancestor_conv_layer,
    )
    n_global = static_objects.props["n_global"]
    n_levels = static_objects.props["n_levels"]
    for ilevel in range(n_levels - 1):
        graph = branchings[ilevel](graph, cond)
        # ### Global
        graph.global_features = dyn_hlvs_layer(
            x=graph.tftx, cond=cond, batch=tree.tbatch_by_level[ilevel + 1]
        )
        assert graph.global_features.shape[1] == n_global
        graph.tftx = ancestor_conv_layer(
            x=graph.tftx,
            cond=cond,
            edge_index=tree.ancestor_ei(ilevel + 1),
            edge_attr=tree.ancestor_ea(ilevel + 1),
            batch=tree.tbatch_by_level[ilevel + 1],
            global_features=graph.global_features,
        )


def test_GlobalFeedBackNN_GINConv(static_objects: DTColl):
    graph, tree, cond, branchings, dyn_hlvs_layer, _ = (
        static_objects.graph,
        static_objects.tree,
        static_objects.cond,
        static_objects.branchings,
        static_objects.dyn_hlvs_layer,
        static_objects.ancestor_conv_layer,
    )
    n_global = static_objects.props["n_global"]
    n_levels = static_objects.props["n_levels"]
    n_features = static_objects.props["n_features"]
    conv = GINConv(
        nn.Sequential(
            nn.Linear(n_features + n_global, n_features),
        )
    )

    for ilevel in range(n_levels - 1):
        graph = branchings[ilevel](graph, cond)
        # ### Global
        global_features = dyn_hlvs_layer(
            x=graph.tftx, cond=cond, batch=tree.tbatch_by_level[ilevel + 1]
        )
        assert global_features.shape[1] == n_global

        graph.tftx = conv(
            x=torch.hstack(
                [graph.tftx, global_features[tree.tbatch_by_level[ilevel + 1]]]
            ),
            edge_index=tree.ancestor_ei(ilevel + 1),
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
    torch.autograd.set_detect_anomaly(True)

    graph, tree, cond, branchings, dyn_hlvs_layer, ancestor_conv_layer = (
        static_objects.graph,
        static_objects.tree,
        static_objects.cond,
        static_objects.branchings,
        static_objects.dyn_hlvs_layer,
        static_objects.ancestor_conv_layer,
    )
    n_global = static_objects.props["n_global"]
    n_levels = static_objects.props["n_levels"]

    tree_lists = branchings[0].tree.tree_lists
    zero_feature = torch.zeros_like(graph.tftx[0])
    x_old = graph.tftx
    for ilevel in range(n_levels - 1):
        graph.global_features = dyn_hlvs_layer(
            x=graph.tftx, cond=cond, batch=tree.tbatch_by_level[ilevel]
        )
        assert graph.global_features.shape[1] == n_global

        graph = branchings[ilevel](graph, cond)
        graph.tftx = ancestor_conv_layer(
            x=graph.tftx,
            cond=cond,
            global_features=graph.global_features,
            edge_index=tree.ancestor_ei(ilevel + 1),
            edge_attr=tree.ancestor_ea(ilevel + 1),
            batch=tree.tbatch_by_level[ilevel + 1],
        )
        leaf = tree_lists[ilevel][0]
        pc_leaf_point = graph.tftx[leaf.idxs[2]]
        sum(pc_leaf_point).backward(retain_graph=True)

        assert x_old.grad is not None
        assert torch.all(x_old.grad[0] == zero_feature)
        assert torch.all(x_old.grad[1] == zero_feature)
        assert torch.any(x_old.grad[2] != zero_feature)


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

    from fgsim.config import conf, defaultconf
    from fgsim.models.gen.gen_deeptree import ModelClass, TreeGraph

    # normalization needs to be set to false, otherwise Batchnorm
    # will propagate some gradient betweeen the events
    device = torch.device("cpu")
    conf.tree.branches = [2, 3, 5]
    conf.tree.features = [128, 64, 32, 3]
    # defaultconf.model_param_options.gen_deeptree.dim_red_in_branching = False
    # defaultconf.model_param_options.gen_deeptree.branching_param.residual = False
    # defaultconf.model_param_options.gen_deeptree.branching_param.final_linear = (
    #     False
    # )
    # defaultconf.model_param_options.gen_deeptree.branching_param.res_mean = False
    # defaultconf.model_param_options.gen_deeptree.branching_param.res_final_layer = (
    #     False
    # )
    conf.ffn.norm = "none"
    defaultconf.model_param_options.gen_deeptree.branching_param.norm = "none"

    model = ModelClass(**defaultconf.model_param_options.gen_deeptree).to(device)

    z = torch.randn(*model.z_shape, requires_grad=True, device=device)
    cond = torch.randn(
        defaultconf.loader.batch_size,
        sum(defaultconf.loader.cond_gen_features),
        requires_grad=True,
        device=device,
    )

    def check_z():
        assert torch.any(z.grad[1] != 0), "Grad not propagated"
        assert torch.all(z.grad[torch.arange(len(z)) != 1] == 0), "Tainted gradient"
        z.grad = None

    def check_cond():
        assert torch.all(
            cond.grad[torch.arange(len(z)) != 1] == 0
        ), "Tainted gradient"
        assert torch.any(cond.grad[1] != 0), "Grad not propagated"
        cond.grad = None

    batch_size = model.batch_size
    features = model.features
    tree = model.tree
    n_levels = len(features)

    # Init the graph object
    graph_tree = TreeGraph(
        tftx=z.reshape(batch_size, features[0]),
        global_features=torch.empty(
            batch_size,
            model.n_global,
            dtype=torch.float,
            device=device,
        ),
        tree=model.tree,
    )
    # check
    cond[[tree.tbatch_by_level[0] == 1]].sum().backward(retain_graph=True)
    check_cond()
    graph_tree.tftx[[tree.tbatch_by_level[0] == 1]].sum().backward(
        retain_graph=True
    )
    check_z()
    batchidx = model.tree.tbatch_by_level[0]
    idxs_level = model.tree.idxs_by_level[0]
    batch_level = batchidx[idxs_level]
    edge_index = model.tree.ancestor_ei(0)
    edge_attr = model.tree.ancestor_ea(0)

    # Do the branching
    for ilevel in range(n_levels - 1):
        # Assign the global features
        ftx_level = graph_tree.tftx_by_level(ilevel)
        graph_tree.global_features = model.prebr_hlvs[ilevel](
            x=ftx_level,
            cond=cond,
            batch=batch_level,
        )
        if graph_tree.global_features.numel():
            graph_tree.global_features[1, :].sum().backward(retain_graph=True)
            check_z()
            graph_tree.global_features[1, :].sum().backward(retain_graph=True)
            check_cond()
        graph_tree = model.branchings[ilevel](graph_tree, cond)

        graph_tree.tftx[[tree.tbatch_by_level[ilevel + 1] == 1]].sum().backward(
            retain_graph=True
        )
        check_z()
        # Assign the new indices for the updated tree
        batchidx = model.tree.tbatch_by_level[ilevel + 1]
        idxs_level = model.tree.idxs_by_level[ilevel + 1]
        batch_level = batchidx[idxs_level]
        edge_index = model.tree.ancestor_ei(ilevel + 1)
        edge_attr = model.tree.ancestor_ea(ilevel + 1)

        graph_tree.tftx = model.ac_mpl[ilevel](
            x=graph_tree.tftx,
            cond=cond,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batchidx,
            global_features=graph_tree.global_features,
        )
        graph_tree.tftx[[tree.tbatch_by_level[ilevel + 1] == 1]].sum().backward(
            retain_graph=True
        )
        check_z()

        if hasattr(model, "child_conv_layers"):
            graph_tree.tftx = model.child_conv_layers[ilevel](
                x=graph_tree.tftx,
                cond=cond,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batchidx,
            )
            graph_tree.tftx[[tree.tbatch_by_level[ilevel + 1] == 1]].sum().backward(
                retain_graph=True
            )
            check_z()

    x = graph_tree.tftx_by_level(-1)
    batchidx = tree.tbatch_by_level[-1][tree.idxs_by_level[-1]]

    x[batchidx == 1].sum().backward(retain_graph=True)

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
    from fgsim.config import conf, defaultconf
    from fgsim.models.gen.gen_deeptree import ModelClass

    conf.ffn.norm = "none"
    conf.ffn.dropout = False
    conf.tree.features[-1] = conf.loader.n_features

    device = torch.device("cpu")
    defaultconf.ffn.norm = "none"
    defaultconf.model_param_options.gen_deeptree.branching_param.norm = "none"
    model = ModelClass(**defaultconf.model_param_options.gen_deeptree).to(device)

    z = torch.randn(*model.z_shape, device=device).requires_grad_()
    cond = (
        torch.ones(
            (
                defaultconf.loader.batch_size,
                sum(defaultconf.loader.cond_gen_features),
            ),
            device=device,
        )
        * conf.loader.n_points
    )
    tftx_copy = z
    n_pointsv = torch.tensor(
        [defaultconf.loader.n_points] * defaultconf.loader.batch_size
    )
    batch = model(z, cond, n_pointsv)
    sum_of_features = batch.x[batch.batch == 1].sum()
    sum_of_features.backward(retain_graph=True)
    zero_feature = torch.zeros_like(tftx_copy[0])
    assert torch.all(tftx_copy.grad[0] == zero_feature)
    assert torch.any(tftx_copy.grad[1] != zero_feature)
    assert torch.all(tftx_copy.grad[2] == zero_feature)
