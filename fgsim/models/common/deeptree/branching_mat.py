import torch

from fgsim.models.common import FFN
from fgsim.utils import check_tensor

from .branching_base import BranchingBase
from .graph_tree import TreeGraph


class BranchingMat(BranchingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        proj_in = self.n_features_source
        if not self.gated_cond:
            proj_in += self.n_global + self.n_cond
        proj_out = self.n_features_source * self.n_branches
        self.proj_nn = FFN(
            proj_in,
            proj_out,
            norm=self.norm,
            bias=False,
            final_linear=self.final_linear or (not self.dim_red and self.lastlayer),
        )

    # Split each of the leafs in the the graph.tree
    # into n_branches and connect them
    def forward(self, graph: TreeGraph, cond) -> TreeGraph:
        batch_size = self.batch_size
        n_branches = self.n_branches
        n_features_target = self.n_features_target
        parents = self.tree.tree_lists[self.level]
        n_parents = len(parents)

        assert graph.cur_level == self.level

        parents_ftxs = graph.tftx[self.tree.idxs_by_level[self.level]].clone()

        # Compute the new feature vectors:
        # for the parents indices generate a matrix where
        # each row is the global vector of the respective event

        parent_global = graph.global_features.repeat(
            self.tree.points_by_level[self.level], 1
        )
        cond_global = cond.repeat(self.tree.points_by_level[self.level], 1)
        # With the idxs of the parent index the event vector

        # The proj_nn projects the (n_parents * n_event) x n_features to a
        # (n_parents * n_event) x (n_features*n_branches) matrix
        # [[parent1], -> [[child1-1 child1-2],
        #   parent2]]     [child2-1 child2-2]]
        if self.gated_cond:
            parents_ftxs += self.cond_GCU(
                parents_ftxs.clone(), torch.hstack([cond_global, parent_global])
            )
            proj_ftx = self.proj_nn(parents_ftxs)
        else:
            proj_ftx = self.proj_nn(
                torch.hstack([parents_ftxs, cond_global, parent_global])
            )
        check_tensor(proj_ftx)
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
        children_ftxs = self.reshape_features(
            proj_ftx,
            n_parents=n_parents,
            batch_size=batch_size,
            n_branches=n_branches,
            n_features=self.n_features_source,
        )
        # assert (
        #     children_ftxs[batch_size]
        #     == proj_ftx[0, self.n_features_source
        # : (self.n_features_source * 2)]
        # ).all()
        del proj_ftx

        children_ftxs = self.add_parent_skip(children_ftxs, parents_ftxs)
        # Do the down projection to the desired dimension
        children_ftxs = self.red_children(children_ftxs)

        check_tensor(children_ftxs)
        graph.tftx = torch.vstack(
            [graph.tftx[..., :n_features_target], children_ftxs]
        )

        graph.cur_level = graph.cur_level + 1
        graph.global_features = graph.global_features
        return graph
