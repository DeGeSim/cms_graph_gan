import torch

from fgsim.models.common import FFN

from .branching_base import BranchingBase
from .graph_tree import TreeGraph


class BranchingEquivar(BranchingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        proj_in = (
            self.n_features_source // self.n_branches + self.n_global + self.n_cond
        )
        proj_out = self.n_features_source
        assert self.n_features_source % self.n_branches == 0
        self.proj_nn = FFN(
            proj_in,
            proj_out,
            norm=self.norm,
            bias=False,
            final_linear=self.final_linear
            and (not self.dim_red and self.lastlayer),
        )
        self.proj_cat = FFN(
            self.n_features_source * 2,
            self.n_features_source,
            norm=self.norm,
            bias=False,
            final_linear=self.final_linear
            and (not self.dim_red and self.lastlayer),
        )
        if self.dim_red:
            self.reduction_nn = FFN(
                self.n_features_source,
                self.n_features_target,
                norm=self.norm,
                bias=False,
                final_linear=self.final_linear or self.lastlayer,
            )

    # Split each of the leafs in the the graph.tree
    # into n_branches and connect them
    def forward(self, graph: TreeGraph, cond) -> TreeGraph:
        batch_size = self.batch_size
        n_branches = self.n_branches
        n_features_source = self.n_features_source
        n_features_target = self.n_features_target
        parents = self.tree.tree_lists[self.level]
        n_parents = len(parents)

        assert graph.cur_level == self.level

        parents_ftxs = graph.tftx[self.tree.idxs_by_level[self.level]]
        parents_ftxs_split = self.reshape_features(
            parents_ftxs,
            n_parents=n_parents,
            batch_size=batch_size,
            n_branches=n_branches,
            n_features=self.n_features_source // n_branches,
        ).reshape(
            batch_size,
            n_parents * n_branches,
            n_features_source // n_branches,
        )

        # Compute the new feature vectors:
        # for the parents indices generate a matrix where
        # each row is the global vector of the respective event

        parent_global = graph.global_features.repeat(
            self.tree.points_by_level[self.level] * n_branches, 1
        ).reshape(
            batch_size,
            n_parents * n_branches,
            -1,
        )
        cond_global = cond.repeat(
            self.tree.points_by_level[self.level] * n_branches, 1
        ).reshape(
            batch_size,
            n_parents * n_branches,
            -1,
        )

        # Project each particle by itself, together with the global and condition
        proj_single = self.proj_nn(
            torch.cat([parents_ftxs_split, cond_global, parent_global], -1)
        )

        # # generate the lists for the equivariant stacking
        proj_aggr = (
            proj_single.reshape(
                batch_size, n_parents, n_branches, n_features_source
            )
            .max(-2)
            .values.unsqueeze(1)
            .repeat(1, 1, n_branches, 1)
            .reshape(batch_size, n_parents * n_branches, n_features_source)
        )

        assert proj_single.shape == proj_aggr.shape
        # proj_single
        children_ftxs = self.proj_cat(torch.cat([proj_single, proj_single], -1))

        # If this branching layer reduces the dimensionality,
        # we need to slice the parent_ftxs for the residual connection
        if not self.dim_red:
            parents_ftxs = parents_ftxs[..., :n_features_target]
        # If residual, add the features of the parent to the children
        if self.residual and (
            self.res_final_layer or self.level + 1 != self.tree.n_levels - 1
        ):
            # this skip connection breaks the equivariance
            # but otherwise we dont have enough feature
            parents_ftxs_full = parents_ftxs.repeat(1, n_branches).reshape(
                batch_size * n_parents, n_branches, n_features_source
            )
            parents_ftxs_full = self.reshape_features(
                parents_ftxs_full,
                n_parents=n_parents,
                batch_size=batch_size,
                n_branches=n_branches,
                n_features=self.n_features_source,
            ).reshape(batch_size, n_parents * n_branches, self.n_features_source)
            children_ftxs += parents_ftxs_full
            # parents_ftxs.reshape(batch_size, n_parents, n_features_source).repeat(
            #     n_branches, 1, 1
            # ).reshape(batch_size, n_parents * n_branches, n_features_source)

            if self.res_mean:
                children_ftxs /= 2

        # model_plotter.save_tensor(
        #     f"branching output level{self.level}",
        #     children_ftxs,
        # )

        # Do the down projection to the desired dimension
        children_ftxs = children_ftxs.reshape(
            batch_size * n_parents * n_branches, n_features_source
        )
        children_ftxs = self.red_children(children_ftxs)
        graph.tftx = torch.vstack(
            [graph.tftx[..., :n_features_target], children_ftxs]
        )

        graph.cur_level = graph.cur_level + 1
        graph.global_features = graph.global_features
        return graph
