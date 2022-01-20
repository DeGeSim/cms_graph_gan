import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from fgsim.config import device
from fgsim.models.subnetworks.gen_deeptree_pc.ancestor_conv import AncestorConvLayer
from fgsim.models.subnetworks.gen_deeptree_pc.branching import BranchingLayer
from fgsim.models.subnetworks.gen_deeptree_pc.dyn_hlvs import DynHLVsLayer
from fgsim.models.subnetworks.gen_deeptree_pc.tree import Node

n_features = 4
n_branches = 2
n_global = 6
n_events = 3


@pytest.fixture
def graph():
    g = Data(
        x=torch.randn(n_events, n_features, dtype=torch.float, device=device),
        edge_index=torch.tensor([[], []], dtype=torch.long, device=device),
        edge_attr=torch.tensor([], dtype=torch.long, device=device),
        event=torch.arange(n_events, dtype=torch.long, device=device),
        tree=[[Node(torch.arange(n_events, dtype=torch.long, device=device))]],
    )
    return g


def test_GlobalFeedBackNN(graph: Data):
    dyn_hlvs_layer = DynHLVsLayer(
        pre_nn=nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
        ),
        post_nn=nn.Sequential(
            nn.Linear(n_features, n_global),
            nn.ReLU(),
        ),
    ).to(device)
    branching_layer = BranchingLayer(
        n_events=n_events,
        n_features=n_features,
        n_branches=n_branches,
        proj_nn=nn.Sequential(
            nn.Linear(n_features + n_global, n_features * n_branches),
            nn.ReLU(),
        ),
    ).to(device)
    ancestor_conv_layer = AncestorConvLayer(
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
    for isplit in range(4):
        # ### Global
        global_features = dyn_hlvs_layer(graph)
        assert global_features.shape[1] == n_global
        graph = branching_layer(graph, global_features)
        graph.x = ancestor_conv_layer(graph, global_features)
