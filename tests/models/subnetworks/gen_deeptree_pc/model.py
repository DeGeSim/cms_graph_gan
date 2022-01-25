import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from fgsim.models.subnetworks.gen_deeptree_pc.ancestor_conv import AncestorConvLayer
from fgsim.models.subnetworks.gen_deeptree_pc.branching import BranchingLayer
from fgsim.models.subnetworks.gen_deeptree_pc.dyn_hlvs import DynHLVsLayer
from fgsim.models.subnetworks.gen_deeptree_pc.tree import Node

n_features = 4
n_branches = 2
n_global = 6
n_events = 3
device = torch.device("cpu")


@pytest.fixture
def graph() -> Data:
    g = Data(
        x=torch.randn(
            n_events,
            n_features,
            dtype=torch.float,
            device=device,
            requires_grad=True,
        ),
        edge_index=torch.tensor([[], []], dtype=torch.long, device=device),
        edge_attr=torch.tensor([], dtype=torch.long, device=device),
        event=torch.arange(n_events, dtype=torch.long, device=device),
        tree=[[Node(torch.arange(n_events, dtype=torch.long, device=device))]],
    )
    return g


@pytest.fixture
def branching_layer() -> BranchingLayer:
    return BranchingLayer(
        n_events=n_events,
        n_features=n_features,
        n_branches=n_branches,
        proj_nn=nn.Sequential(
            nn.Linear(n_features + n_global, n_features * n_branches),
            nn.ReLU(),
        ),
    ).to(device)


@pytest.fixture
def dyn_hlvs_layer() -> DynHLVsLayer:
    return DynHLVsLayer(
        pre_nn=nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
        ),
        post_nn=nn.Sequential(
            nn.Linear(n_features, n_global),
            nn.ReLU(),
        ),
    ).to(device)


@pytest.fixture
def ancestor_conv_layer() -> AncestorConvLayer:
    return AncestorConvLayer(
        msg_gen=nn.Sequential(
            nn.Linear(n_features + n_global, n_features),
            nn.ReLU(),
        ),
        update_nn=nn.Sequential(
            # agreegated features + previous feature vector + global
            nn.Linear(2 * n_features + n_global, n_features),
            nn.ReLU(),
        ),
    ).to(device)


def test_GlobalFeedBackNN(
    graph: Data,
    branching_layer: BranchingLayer,
    dyn_hlvs_layer: DynHLVsLayer,
    ancestor_conv_layer: AncestorConvLayer,
):
    for isplit in range(4):
        # ### Global
        global_features = dyn_hlvs_layer(graph)
        assert global_features.shape[1] == n_global
        graph = branching_layer(graph, global_features)
        graph.x = ancestor_conv_layer(graph, global_features)


def test_full_NN_compute_graph(
    graph: Data,
    branching_layer: BranchingLayer,
    dyn_hlvs_layer: DynHLVsLayer,
    ancestor_conv_layer: AncestorConvLayer,
):
    """
    Make sure that the events are independent.
    For this, we apply branching and make sure, that the gradient only
    is nonzero for the root of the event we apply the `backwards()` on.
    Args:
      graph (Data): The original graph.
      branching_layer (BranchingLayer): The branching layer to test.
      global_features (torch.Tensor): torch.Tensor
    """
    zero_feature = torch.zeros_like(graph.x[0])
    x_old = graph.x
    for isplit in range(4):
        # ### Global
        global_features = dyn_hlvs_layer(graph)
        assert global_features.shape[1] == n_global
        graph = branching_layer(graph, global_features)
        graph.x = ancestor_conv_layer(graph, global_features)

        leaf = graph.tree[-1][0]
        pc_leaf_point = graph.x[leaf.idxs[2]]
        sum(pc_leaf_point).backward(retain_graph=True)

        assert x_old.grad is not None
        assert torch.all(x_old.grad[0] == zero_feature)
        assert torch.all(x_old.grad[1] == zero_feature)
        assert torch.any(x_old.grad[2] != zero_feature)
