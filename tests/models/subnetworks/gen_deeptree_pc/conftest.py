from dataclasses import dataclass
from itertools import product
from typing import Dict, List

import pytest
import torch

from fgsim.models.common import FFN, DynHLVsLayer
from fgsim.models.common.deeptree import (
    BranchingLayer,
    DeepConv,
    GraphTreeWrapper,
    Tree,
    TreeGenType,
)

device = torch.device("cpu")


@dataclass
class DTColl:
    props: Dict[str, int]
    graph: GraphTreeWrapper
    tree: Tree
    branching_layers: List[BranchingLayer]
    dyn_hlvs_layer: DynHLVsLayer
    ancestor_conv_layer: DeepConv


def object_gen(props: Dict[str, int]) -> DTColl:
    n_features = props["n_features"]
    n_branches = props["n_branches"]
    n_global = props["n_global"]
    batch_size = props["batch_size"]
    n_levels = props["n_levels"]

    features = [n_features for _ in range(n_levels)]
    branches = [n_branches for _ in range(n_levels - 1)]

    graph = GraphTreeWrapper(
        TreeGenType(
            torch.randn(
                batch_size,
                n_features,
                dtype=torch.float,
                device=device,
                requires_grad=True,
            ),
            batch_size=batch_size,
        )
    )

    tree = Tree(
        branches=branches,
        features=features,
        batch_size=batch_size,
        device=device,
    )
    branching_layers = [
        BranchingLayer(
            tree=tree,
            level=level,
            n_global=n_global,
            residual=False,
        ).to(device)
        for level in range(n_levels - 1)
    ]

    dyn_hlvs_layer = DynHLVsLayer(
        n_features=n_features,
        batch_size=batch_size,
        n_global=n_global,
        device=device,
    )

    ancestor_conv_layer = DeepConv(
        in_features=n_features,
        out_features=n_features,
        n_global=n_global,
        residual=False,
    )

    # BatchNorm makes the batches dependent
    # If we want to check the gradients for independence,
    # we need to disable batchnorm
    def overwrite_ffn_without_batchnorm(ow_nn):
        assert isinstance(ow_nn, FFN)
        ow_nn = FFN(ow_nn.input_dim, ow_nn.output_dim, normalize=False)
        return ow_nn

    for brl in branching_layers:
        brl.proj_nn = overwrite_ffn_without_batchnorm(brl.proj_nn)

    dyn_hlvs_layer.pre_nn = overwrite_ffn_without_batchnorm(dyn_hlvs_layer.pre_nn)
    dyn_hlvs_layer.post_nn = overwrite_ffn_without_batchnorm(dyn_hlvs_layer.post_nn)

    ancestor_conv_layer.msg_nn = overwrite_ffn_without_batchnorm(
        ancestor_conv_layer.msg_nn
    )
    ancestor_conv_layer.update_nn = overwrite_ffn_without_batchnorm(
        ancestor_conv_layer.update_nn
    )

    return DTColl(
        props=props,
        graph=graph,
        tree=tree,
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
        "batch_size": 3,
        "n_levels": 4,
    }
    return props


@pytest.fixture(params=product([2], [1, 2], [3], [2], [2, 4]))
def dyn_props(request):
    n_features, n_branches, batch_size, n_global, n_levels = request.param
    props = {
        "n_features": n_features,
        "n_branches": n_branches,
        "batch_size": batch_size,
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
