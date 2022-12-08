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

    graph = static_objects.graph
    branching_layers = static_objects.branching_layers
    dyn_hlvs_layer = static_objects.dyn_hlvs_layer
    tree = static_objects.tree

    new_graph1 = branching_layers[0](graph)
    new_global_features = dyn_hlvs_layer(
        x=new_graph1.tftx, cond=graph.cond, batch=tree.tbatch_by_level[1]
    )
    event_2_global = new_global_features[2]
    sum(event_2_global).backward(retain_graph=True)

    zero_feature = torch.zeros_like(graph.tftx[0])
    assert graph.tftx.grad is not None
    assert torch.all(graph.tftx.grad[0] == zero_feature)
    assert torch.all(graph.tftx.grad[1] == zero_feature)
    assert torch.any(graph.tftx.grad[2] != zero_feature)


def test_dyn_hlvs_compute_graph2(static_objects: DTColl):
    graph = static_objects.graph
    dyn_hlvs_layer = static_objects.dyn_hlvs_layer
    tree = static_objects.tree

    new_global_features = dyn_hlvs_layer(
        x=graph.tftx, cond=graph.cond, batch=tree.tbatch_by_level[0]
    )
    event_2_global = new_global_features[2]
    sum(event_2_global).backward(retain_graph=True)

    zero_feature = torch.zeros_like(graph.tftx[0])
    assert graph.tftx.grad is not None
    assert torch.all(graph.tftx.grad[0] == zero_feature)
    assert torch.all(graph.tftx.grad[1] == zero_feature)
    assert torch.any(graph.tftx.grad[2] != zero_feature)
