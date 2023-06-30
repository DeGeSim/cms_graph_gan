import torch
import torch.nn as nn

from fgsim.models.common import FFN
from fgsim.models.pool.bppool import BipartPool
from fgsim.utils import check_tensor

from .branching_base import BranchingBase
from .graph_tree import TreeGraph


class BranchingBPPool(BranchingBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # proj_in = self.n_features_source + self.n_global + self.n_cond
        # proj_out = self.n_features_source * self.n_branches

        self.n_heads = 8
        self.pool = BipartPool(
            in_channels=self.n_features_source,
            ratio=self.n_branches,
            n_heads=self.n_heads,
            mode="mpl",
            batch_size=self.batch_size,
        )
        self.lin_in = nn.Linear(
            self.n_features_source + self.n_cond + self.n_global,
            self.n_features_source * self.n_heads,
        )
        self.lin_out = nn.Linear(
            self.n_features_source * self.n_heads,
            self.n_features_source,
        )
        attbatch_idx = torch.arange(self.batch_size * self.n_parents)
        self.register_buffer("attbatch_idx", attbatch_idx, persistent=True)

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
        n_features_target = self.n_features_target
        parents = self.tree.tree_lists[self.level]
        n_parents = len(parents)

        assert graph.cur_level == self.level

        # parentidx > batchsize > n_features_source
        treeidxs = self.tree.idxs_by_level[self.level]
        parents_ftxs = graph.tftx[treeidxs]
        self.tree.tbatch_by_level[self.level][treeidxs]

        # Compute the new feature vectors:
        # for the parents indices generate a matrix where
        # each row is the global vector of the respective event

        parent_global = graph.global_features.repeat(
            self.tree.points_by_level[self.level], 1
        )
        cond_global = cond.repeat(self.tree.points_by_level[self.level], 1)

        pstack = torch.hstack(
            [
                parents_ftxs,
                cond_global,
                parent_global,
            ]
        )

        pstacktf = self.lin_in(pstack)
        att_out, _cbatchidx = self.pool(x=pstacktf, batch=self.attbatch_idx)
        proj_ftx = self.lin_out(att_out)
        # children_ftxs = children_ftxs.reshape(
        #     batch_size * n_parents * n_branches, self.n_features_source
        # )
        children_ftxs = self.reshape_features(
            proj_ftx,
            n_parents=n_parents,
            batch_size=batch_size,
            n_branches=n_branches,
            n_features=self.n_features_source,
        )

        children_ftxs = self.add_parent_skip(children_ftxs, parents_ftxs)

        children_ftxs = self.red_children(children_ftxs)

        check_tensor(children_ftxs)
        graph.tftx = torch.vstack(
            [graph.tftx[..., :n_features_target], children_ftxs]
        )

        graph.cur_level = graph.cur_level + 1
        graph.global_features = graph.global_features
        return graph
