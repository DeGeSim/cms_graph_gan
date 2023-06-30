from dataclasses import dataclass
from itertools import product
from typing import Dict, List

import pytest
import torch
from torch import nn

# install_import_hook("fgsim")
from fgsim.models.common import FFN, DynHLVsLayer
from fgsim.models.common.deeptree import BranchingLayer, DeepConv, Tree, TreeGraph

# from typeguard.importhook import install_import_hook


device = torch.device("cpu")


@dataclass
class DTColl:
    props: Dict[str, int]
    graph: TreeGraph
    tree: Tree
    branchings: List[BranchingLayer]
    dyn_hlvs_layer: DynHLVsLayer
    ancestor_conv_layer: DeepConv
    cond: torch.Tensor


def object_gen(props: Dict[str, int]) -> DTColl:
    n_features = props["n_features"]
    n_branches = props["n_branches"]
    n_global = props["n_global"]
    n_cond = props["n_cond"]
    batch_size = props["batch_size"]
    n_levels = props["n_levels"]

    features = [n_features for _ in range(n_levels)]
    branches = [n_branches for _ in range(n_levels - 1)]
    tree = Tree(
        branches=branches,
        features=features,
        batch_size=batch_size,
        connect_all_ancestors=True,
    )
    global_features = torch.randn(
        batch_size, n_global, dtype=torch.float, device=device
    )
    cond = torch.randn(
        batch_size,
        n_cond,
        dtype=torch.float,
        device=device,
        requires_grad=True,
    )
    graph = TreeGraph(
        tftx=torch.randn(
            batch_size,
            n_features,
            dtype=torch.float,
            device=device,
            requires_grad=True,
        ),
        tree=tree,
        global_features=global_features,
    )

    branchings = [
        BranchingLayer(
            tree=tree,
            level=level,
            n_global=n_global,
            n_cond=n_cond,
            dim_red=False,
            final_linear=True,
            norm="none",
            residual=False,
            res_final_layer=False,
            res_mean=False,
            dim_red_skip=True,
            mode="mat",
        ).to(device)
        for level in range(n_levels - 1)
    ]

    dyn_hlvs_layer = DynHLVsLayer(
        n_features=n_features,
        batch_size=batch_size,
        n_global=n_global,
        n_cond=n_cond,
    )

    ancestor_conv_layer = DeepConv(
        in_features=n_features,
        out_features=n_features,
        nns="both",
        n_global=n_global,
        n_cond=n_cond,
        add_self_loops=True,
        msg_nn_include_edge_attr=True,
        msg_nn_include_global=False,
        upd_nn_include_global=True,
        msg_nn_final_linear=True,
        upd_nn_final_linear=True,
        residual=True,
    )

    # BatchNorm makes the batches dependent
    # If we want to check the gradients for independence,
    # we need to disable batchnorm

    for brl in branchings:
        recur_remove_ffnorm(brl)
    recur_remove_ffnorm(dyn_hlvs_layer)
    recur_remove_ffnorm(ancestor_conv_layer)

    return DTColl(
        props=props,
        graph=graph,
        tree=tree,
        branchings=branchings,
        dyn_hlvs_layer=dyn_hlvs_layer,
        ancestor_conv_layer=ancestor_conv_layer,
        cond=cond,
    )


@pytest.fixture()
def static_props():
    props = {
        "n_features": 4,
        "n_branches": 2,
        "n_global": 1,
        "n_cond": 1,
        "batch_size": 3,
        "n_levels": 4,
    }
    return props


@pytest.fixture(params=product([1, 2], [2, 4]))
def dyn_props(request):
    n_branches, n_levels = request.param
    props = {
        "n_features": 2,
        "n_branches": n_branches,
        "batch_size": 3,
        "n_global": 2,
        "n_cond": 1,
        "n_levels": n_levels,
    }
    return props


@pytest.fixture(
    params=product(
        ["mat", "equivar", "noise", "bppool"], [False, True], [False, True]
    )
)
def branching_objects(request, static_props):
    (mode, dim_red, residual) = request.param
    objs = object_gen(static_props)
    objs.branchings = [
        BranchingLayer(
            tree=objs.tree,
            level=level,
            n_global=static_props["n_global"],
            n_cond=static_props["n_cond"],
            dim_red=dim_red,
            dim_red_skip=True,
            final_linear=True,
            norm="none",
            residual=residual,
            res_final_layer=False,
            res_mean=False,
            mode=mode,
        ).to(device)
        for level in range(static_props["n_levels"] - 1)
    ]
    recur_remove_ffnorm(objs.branchings)
    return objs


@pytest.fixture()
def static_objects(static_props):
    return object_gen(static_props)


@pytest.fixture()
def dyn_objects(dyn_props):
    return object_gen(dyn_props)


# remove batchnorm for testing to avoid grradient contamination
def recur_remove_ffnorm(m):
    if isinstance(m, list):
        for sm in m:
            recur_remove_ffnorm(sm)
    if isinstance(m, FFN):
        m.seq = nn.Sequential(
            *[sm for sm in m.seq if type(sm) not in [nn.BatchNorm1d, nn.LayerNorm]]
        )
    elif isinstance(m, nn.Module):
        for sm in m.children():
            recur_remove_ffnorm(sm)
    return
