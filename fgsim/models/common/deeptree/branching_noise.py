import torch

from fgsim.models.common import FFN

from .branching_base import BranchingBase
from .graph_tree import TreeGraph


class BranchingNoise(BranchingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_noise = min(max(self.n_features_source, 5), 15)
        proj_in = (
            self.n_features_source + self.n_global + self.n_cond + self.n_noise
        )
        proj_out = self.n_features_source
        self.proj_nn = FFN(
            proj_in,
            proj_out,
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
        parents_ftxs_split = parents_ftxs.repeat(1, n_branches).reshape(
            batch_size * n_parents, n_branches, n_features_source
        )
        parents_ftxs_split = self.reshape_features(
            parents_ftxs_split,
            n_parents=n_parents,
            batch_size=batch_size,
            n_branches=n_branches,
            n_features=self.n_features_source,
        )

        parent_global = graph.global_features.repeat(
            n_parents * n_branches, 1
        ).reshape(batch_size * n_parents * n_branches, -1)
        cond_global = cond.repeat(n_parents * n_branches, 1).reshape(
            batch_size * n_parents * n_branches, -1
        )
        noise = torch.rand(
            batch_size * n_parents * n_branches,
            self.n_noise,
            device=graph.tftx.device,
        )

        # Project each particle by itself, together with the global and condition
        proj_ftx = self.proj_nn(
            torch.cat([parents_ftxs_split, cond_global, parent_global, noise], -1)
        )
        children_ftxs = proj_ftx

        # If this branching layer reduces the dimensionality,
        # we need to slice the parent_ftxs for the residual connection
        if not self.dim_red:
            parents_ftxs_split = parents_ftxs_split[..., :n_features_target]
        # If residual, add the features of the parent to the children
        if self.residual and (
            self.res_final_layer or self.level + 1 != self.tree.n_levels - 1
        ):
            children_ftxs += parents_ftxs_split
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
