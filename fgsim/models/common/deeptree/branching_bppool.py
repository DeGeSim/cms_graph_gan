import torch
import torch.nn as nn

from fgsim.models.common import FFN
from fgsim.models.pool.bppool import BipartPool
from fgsim.plot.model_plotter import model_plotter
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
        self.in_nn = FFN(
            self.n_features_source + self.n_cond + self.n_global,
            self.n_features_source * self.n_heads,
        )
        # self.out_nn = FFN(
        #     self.n_features_source * self.n_heads,
        #     self.n_features_source,
        # )
        self.out_seq = nn.Sequential(
            nn.Linear(
                self.n_features_source * self.n_heads,
                self.n_features_source,
            ),
            nn.BatchNorm1d(self.n_features_source),
        )
        # self.lin_in = nn.Linear(
        #     self.n_features_source + self.n_cond + self.n_global,
        #     self.n_features_source * self.n_heads,
        # )
        # self.bn_in = nn.BatchNorm1d(
        #     self.n_features_source * self.n_heads,
        #     affine=False,
        #     track_running_stats=False,
        # )
        # self.lin_out = nn.Linear(
        #     self.n_features_source * self.n_heads,
        #     self.n_features_source,
        # )
        # self.bn_out = nn.BatchNorm1d(
        #     self.n_features_source, affine=False, track_running_stats=False
        # )
        attbatch_idx = torch.arange(self.batch_size * self.n_parents)
        self.register_buffer("attbatch_idx", attbatch_idx, persistent=True)

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
        parents_ftxs = graph.tftx[treeidxs[0] : treeidxs[-1] + 1]
        self.tree.tbatch_by_level[self.level][treeidxs]

        # model_plotter.save_tensor(
        #     f"branching input level{self.level}",
        #     parents_ftxs,
        # )

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

        pstacktf = self.in_nn(pstack)
        model_plotter.save_tensor(
            f"branching level{self.level} in_nn output",
            pstacktf,
        )
        att_out, _cbatchidx = self.pool(x=pstacktf, batch=self.attbatch_idx)
        att_out = att_out.reshape(
            n_parents * batch_size * n_branches,
            self.n_features_source * self.n_heads,
        )
        # model_plotter.save_tensor(
        #     f"branching level{self.level} att_out branching output",
        #     att_out,
        # )

        children_ftxs = self.reshape_features(
            att_out,
            n_parents=n_parents,
            batch_size=batch_size,
            n_branches=n_branches,
            n_features=self.n_features_source * self.n_heads,
        )
        # model_plotter.save_tensor(
        #     f"branching level{self.level} reshaped branching output",
        #     children_ftxs,
        # )

        children_ftxs = self.out_seq(children_ftxs)
        # model_plotter.save_tensor(
        #     f"branching level{self.level} out_nn output",
        #     children_ftxs,
        # )
        # model_plotter.plot_model_outputs().savefig("wd/act.pdf")
        # exit()

        children_ftxs = self.add_parent_skip(children_ftxs, parents_ftxs)
        # model_plotter.save_tensor(
        #     f"branching level{self.level} parent skip output",
        #     children_ftxs,
        # )

        children_ftxs = self.red_children(children_ftxs)
        # model_plotter.save_tensor(
        #     f"branching level{self.level} dim red output",
        #     children_ftxs,
        # )
        check_tensor(children_ftxs)

        graph.tftx = torch.vstack(
            [graph.tftx[..., :n_features_target], children_ftxs]
        )

        graph.cur_level = graph.cur_level + 1
        graph.global_features = graph.global_features
        return graph
