from math import prod

import torch
import torch.nn as nn

from fgsim.models.common import FFN

from .graph_tree import GraphTreeWrapper, TreeGenType
from .tree import Tree


class BranchingLayer(nn.Module):
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
        self.res_mean = res_mean
        self.res_final_layer = res_final_layer
        self.n_branches = self.tree.branches[level]
        self.n_features_source = self.tree.features[level]
        self.n_features_target = self.tree.features[level + int(self.dim_red)]

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

        lastlayer = level + 1 == len(tree.features) - 1

        self.proj_nn = FFN(
            self.n_features_source + n_global + n_cond,
            self.n_features_source * self.n_branches,
            norm=self.norm,
            bias=False,
            final_linear=self.final_linear or (not self.dim_red and lastlayer),
        )
        if self.dim_red:
            self.reduction_nn = FFN(
                self.n_features_source,
                self.n_features_target,
                norm=self.norm,
                bias=False,
                final_linear=self.final_linear or lastlayer,
            )

    # Split each of the leafs in the the graph.tree into n_branches and connect them
    def forward(self, graph: GraphTreeWrapper) -> GraphTreeWrapper:
        batch_size = self.batch_size
        n_branches = self.n_branches
        n_features_target = self.n_features_target
        parents = self.tree.tree_lists[self.level]
        n_parents = len(parents)
        assert graph.cur_level == self.level

        parents_ftxs = graph.tftx[self.tree.idxs_by_level[self.level]]
        device = parents_ftxs.device

        # Compute the new feature vectors:
        parents_idxs = torch.cat([parent.idxs for parent in parents])
        assert (parents_idxs == self.tree.parents_idxs_by_level[self.level]).all()
        # for the parents indices generate a matrix where
        # each row is the global vector of the respective event
        if graph.global_features.numel() == 0:
            graph.global_features = torch.empty(
                batch_size, self.n_global, dtype=torch.float, device=device
            )
        parent_global = graph.global_features[parents_idxs % batch_size, :]
        cond_global = graph.cond[parents_idxs % batch_size, :]
        # With the idxs of the parent index the event vector

        # The proj_nn projects the (n_parents * n_event) x n_features to a
        # (n_parents * n_event) x (n_features*n_branches) matrix
        # [[parent1], -> [[child1-1 child1-2],
        #   parent2]]     [child2-1 child2-2]]
        proj_ftx = self.proj_nn(
            torch.hstack([parents_ftxs, cond_global, parent_global])
        )

        assert parents_ftxs.shape[-1] == self.n_features_source
        assert proj_ftx.shape[-1] == self.n_features_source * self.n_branches
        assert proj_ftx.shape == (
            n_parents * batch_size,
            n_branches * self.n_features_source,
        )

        # reshape the projected
        # for a single batch
        # [[child1-1 child1-2], -> [[child1-1,
        #  [child2-1 child2-2]]      child1-2,
        #                            child2-1,
        #                            child1-2]]
        children_ftxs = reshape_features(
            proj_ftx,
            n_parents=n_parents,
            batch_size=batch_size,
            n_branches=n_branches,
            n_features=self.n_features_source,
        )
        # assert (
        #     children_ftxs[batch_size]
        #     == proj_ftx[0, self.n_features_source : (self.n_features_source * 2)]
        # ).all()
        del proj_ftx

        # If this branching layer reduces the dimensionality, we need to slice the
        # parent_ftxs for the residual connection
        if not self.dim_red:
            parents_ftxs = parents_ftxs[..., :n_features_target]
        # If residual, add the features of the parent to the children
        if self.residual and (
            self.res_final_layer or self.level + 1 != self.tree.n_levels - 1
        ):
            children_ftxs += parents_ftxs.repeat(n_branches, 1)
            if self.res_mean:
                children_ftxs /= 2

        # Do the down projection to the desired dimension
        if self.dim_red:
            children_ftxs = self.reduction_nn(children_ftxs)

        # graph.data.tftx = torch.vstack(
        #     [graph.tftx[..., :n_features_target], children_ftxs]
        # )
        # graph.data.cond = graph.cond
        # graph.data.idxs_by_level = self.tree.idxs_by_level + [level_idx]
        # graph.data.cur_level = graph.cur_level + 1
        # graph.data.batch_size = batch_size
        # graph.data.global_features = graph.global_features
        # graph.data.tbatch = torch.arange(
        #     batch_size, dtype=torch.long, device=device
        # ).repeat((len(graph.data.tftx)) // batch_size)
        # return graph

        return GraphTreeWrapper(
            TreeGenType(
                tftx=torch.vstack(
                    [graph.tftx[..., :n_features_target], children_ftxs]
                ),
                cond=graph.cond,
                cur_level=graph.cur_level + 1,
                batch_size=batch_size,
                global_features=graph.global_features,
            ),
            tree=graph.tree,
        )


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
