from math import prod

import torch
import torch.nn as nn

from fgsim.utils import check_tensor

from .tree import Tree

# from fgsim.plot.model_plotter import model_plotter


class BranchingBase(nn.Module):
    """Splits the last set of Nodes of a given graph.
    Order for x : node>event>branch
    Example with 3 events,2 branches for a single split:
    FeatureIndex[I]/Event[E]/Branch[B]
    F|E|B
    -|-|-
    0|0|0
    1|1|0
    2|2|0
    3|0|0
    4|0|1
    5|1|0
    6|1|1
    7|2|0
    8|2|1
    """

    def __init__(
        self,
        tree: Tree,
        n_global: int,
        level: int,
        n_cond: int,
        residual: bool,
        final_linear: bool,
        norm: str,
        dim_red: bool,
        res_mean: bool,
        res_final_layer: bool,
        dim_red_skip: bool,
    ):
        super().__init__()
        assert 0 <= level < len(tree.features)
        self.tree = tree
        self.batch_size = self.tree.batch_size
        self.n_global = n_global
        self.n_cond = n_cond
        self.level = level
        self.residual = residual
        self.final_linear = final_linear
        self.norm = norm
        self.dim_red = dim_red
        self.dim_red_skip = dim_red_skip
        self.res_mean = res_mean
        self.res_final_layer = res_final_layer
        self.n_branches = self.tree.branches[level]
        self.n_features_source = self.tree.features[level]
        self.n_features_target = self.tree.features[level + int(self.dim_red)]
        self.parents = self.tree.tree_lists[self.level]
        self.n_parents = len(self.parents)

        # Calculate the number of nodes currently in the graphs
        self.points = prod([br for br in self.tree.branches[: self.level]])
        assert self.points == self.tree.points_by_level[self.level]

        if res_mean or res_final_layer:
            assert residual
        # if residual:
        #     assert final_linear

        if self.dim_red:
            assert self.n_features_source >= self.n_features_target
        else:
            assert self.n_features_source == self.n_features_target

        self.lastlayer = level + 1 == len(tree.features) - 1

    def reshape_features(self, *args, **kwargs):
        return reshape_features(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def red_children(self, children_ftxs: torch.Tensor) -> torch.Tensor:
        if self.dim_red:
            children_ftxs_red = self.reduction_nn(children_ftxs)
            if self.dim_red_skip:
                children_ftxs_red += children_ftxs[
                    ..., : self.n_features_target
                ].clone()
            children_ftxs = children_ftxs_red
        check_tensor(children_ftxs)
        return children_ftxs


@torch.jit.script
def reshape_features(
    mtx: torch.Tensor,
    n_parents: int,
    batch_size: int,
    n_branches: int,
    n_features: int,
):
    return (
        # batch_size*n_parents, n_branches * n_features
        mtx.reshape(n_parents, batch_size, n_branches * n_features)
        .transpose(1, 2)  # n_parents, n_branches * n_features, batch_size
        .reshape(n_parents * n_branches, n_features, batch_size)
        .transpose(1, 2)  # n_parents * n_branches, batch_size, n_features
        .reshape(n_parents * n_branches * batch_size, n_features)
    )
