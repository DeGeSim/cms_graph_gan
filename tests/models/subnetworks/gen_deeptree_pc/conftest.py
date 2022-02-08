from dataclasses import dataclass
from itertools import product
from typing import Dict

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from fgsim.models.subnetworks.gen_deeptree_pc.ancestor_conv import AncestorConv
from fgsim.models.subnetworks.gen_deeptree_pc.branching import BranchingLayer
from fgsim.models.subnetworks.gen_deeptree_pc.dyn_hlvs import DynHLVsLayer

device = torch.device("cpu")


@dataclass
class DTColl:
    props: Dict[str, int]
    graph: Data
    global_features: torch.Tensor
    branching_layer: BranchingLayer
    dyn_hlvs_layer: DynHLVsLayer
    ancestor_conv_layer: AncestorConv


def object_gen(props: Dict[str, int]) -> DTColl:
    n_features = props["n_features"]
    n_branches = props["n_branches"]
    n_global = props["n_global"]
    n_events = props["n_events"]
    n_levels = props["n_levels"]
    graph = Data(
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
    )

    global_features = torch.randn(
        n_events,
        n_global,
        dtype=torch.float,
        device=device,
    )

    branching_layer = BranchingLayer(
        n_levels=n_levels,
        n_events=n_events,
        n_features=n_features,
        n_branches=n_branches,
        proj_nn=nn.Sequential(
            nn.Linear(n_features + n_global, n_features * n_branches),
            nn.ReLU(),
        ),
        device=device,
    ).to(device)

    dyn_hlvs_layer = DynHLVsLayer(
        pre_nn=nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
        ),
        post_nn=nn.Sequential(
            nn.Linear(n_features * 2, n_global),
            nn.ReLU(),
        ),
        n_events=n_events,
    ).to(device)

    ancestor_conv_layer = AncestorConv(
        msg_gen=nn.Sequential(
            nn.Linear(n_features + n_global + 1, n_features),
            nn.ReLU(),
        ),
        update_nn=nn.Sequential(
            # agreegated features + previous feature vector + global
            nn.Linear(2 * n_features + n_global, n_features),
            nn.ReLU(),
        ),
    ).to(device)
    return DTColl(
        props=props,
        graph=graph,
        global_features=global_features,
        branching_layer=branching_layer,
        dyn_hlvs_layer=dyn_hlvs_layer,
        ancestor_conv_layer=ancestor_conv_layer,
    )


@pytest.fixture()
def static_props():
    props = {
        "n_features": 4,
        "n_branches": 2,
        "n_global": 6,
        "n_events": 3,
        "n_levels": 4,
    }
    return props


@pytest.fixture(params=product([2], [1, 2], [3], [2], [1, 4]))
def dyn_props(request):
    n_features, n_branches, n_events, n_global, n_levels = request.param
    props = {
        "n_features": n_features,
        "n_branches": n_branches,
        "n_events": n_events,
        "n_global": n_global,
        "n_levels": n_levels,
    }
    return props


@pytest.fixture()
def static_objects(static_props):
    return object_gen(static_props)


@pytest.fixture()
def dyn_objects(dyn_props):
    return object_gen(dyn_props)
