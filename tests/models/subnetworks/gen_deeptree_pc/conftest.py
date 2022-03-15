from dataclasses import dataclass
from itertools import product
from typing import Dict, List

import pytest
import torch
from torch_geometric.data import Data

from fgsim.models.branching.branching import BranchingLayer
from fgsim.models.branching.tree import Tree
from fgsim.models.layer.ancestor_conv import AncestorConv
from fgsim.models.pooling.dyn_hlvs import DynHLVsLayer

device = torch.device("cpu")


@dataclass
class DTColl:
    props: Dict[str, int]
    graph: Data
    branching_layers: List[BranchingLayer]
    dyn_hlvs_layer: DynHLVsLayer
    ancestor_conv_layer: AncestorConv


def object_gen(props: Dict[str, int]) -> DTColl:
    n_features = props["n_features"]
    n_branches = props["n_branches"]
    n_global = props["n_global"]
    n_events = props["n_events"]
    n_levels = props["n_levels"]

    # features = [n_features for _ in range(n_levels)]
    branches = [0] + [n_branches for _ in range(n_levels - 1)]

    graph = Data(
        x=torch.randn(
            n_events,
            n_features,
            dtype=torch.float,
            device=device,
            requires_grad=True,
        ),
        edge_index=torch.empty(2, 0, dtype=torch.long, device=device),
        edge_attr=torch.empty(0, 1, dtype=torch.float, device=device),
        event=torch.arange(n_events, dtype=torch.long, device=device),
        global_features=torch.randn(
            n_events,
            n_global,
            dtype=torch.float,
            device=device,
        ),
    )

    tree = Tree(
        branches=branches,
        n_events=n_events,
        device=device,
    )
    branching_layers = [
        BranchingLayer(
            tree=tree, level=level, n_features=n_features, n_global=n_global
        ).to(device)
        for level in range(1, n_levels)
    ]

    dyn_hlvs_layer = DynHLVsLayer(
        n_features=n_features,
        n_events=n_events,
        n_global=n_global,
        device=device,
    )

    ancestor_conv_layer = AncestorConv(
        in_features=n_features,
        out_features=n_features,
        n_global=n_global,
    ).to(device)
    return DTColl(
        props=props,
        graph=graph,
        branching_layers=branching_layers,
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


@pytest.fixture(params=product([2], [1, 2], [3], [2], [2, 4]))
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
