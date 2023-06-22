import torch
from conftest import DTColl

device = torch.device("cpu")


def test_dyn_hlvs_compute_graph(static_objects: DTColl):
    """
    Make sure that the events are independent.
    For this, we apply branching and make sure, that the gradient only
    is nonzero for the root of the event we apply the `backwards()` on.
    Args:
      graph (Data): The original graph.
      branching_layer (BranchingLayer): The branching layer to test.
      global_features (torch.Tensor): torch.Tensor
    """
    graph, tree, cond, branchings, dyn_hlvs_layer = (
        static_objects.graph,
        static_objects.tree,
        static_objects.cond,
        static_objects.branchings,
        static_objects.dyn_hlvs_layer,
    )
    tftx_copy = graph.tftx.requires_grad_()

    new_graph1 = branchings[0](graph, cond)
    new_global_features = dyn_hlvs_layer(
        x=new_graph1.tftx, cond=cond, batch=tree.tbatch_by_level[1]
    )
    event_2_global = new_global_features[2]
    sum(event_2_global).backward(retain_graph=True)

    zero_feature = torch.zeros_like(graph.tftx[0])
    assert tftx_copy.grad is not None
    assert torch.all(tftx_copy.grad[0] == zero_feature)
    assert torch.all(tftx_copy.grad[1] == zero_feature)
    assert torch.any(tftx_copy.grad[2] != zero_feature)


def test_dyn_hlvs_compute_graph2(static_objects: DTColl):
    graph, tree, cond, _, dyn_hlvs_layer = (
        static_objects.graph,
        static_objects.tree,
        static_objects.cond,
        static_objects.branchings,
        static_objects.dyn_hlvs_layer,
    )
    tftx_copy = graph.tftx.requires_grad_()

    new_global_features = dyn_hlvs_layer(
        x=graph.tftx, cond=cond, batch=tree.tbatch_by_level[0]
    )
    event_2_global = new_global_features[2]
    sum(event_2_global).backward(retain_graph=True)

    zero_feature = torch.zeros_like(graph.tftx[0])
    assert tftx_copy.grad is not None
    assert torch.all(tftx_copy.grad[0] == zero_feature)
    assert torch.all(tftx_copy.grad[1] == zero_feature)
    assert torch.any(tftx_copy.grad[2] != zero_feature)
