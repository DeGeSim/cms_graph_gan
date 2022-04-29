import torch
from torch_geometric.data import Data

from fgsim.config import conf
from fgsim.models.branching.graph_tree import GraphTreeWrapper

from .disc_graphgym import ModelClass as LevelDisc


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.branches = conf.tree.branches
        self.features = conf.tree.features
        self.x_by_level = len(conf.tree.features)

        self.level_discs = torch.nn.ModuleList(
            [LevelDisc() for _ in range(self.x_by_level)]
        )
        # self.branching_discs = torch.nn.ModuleList(
        #     [BranchingDisc() for _ in range(self.x_by_level - 1)]
        # )

    def forward(self, batch: Data):
        batch = GraphTreeWrapper(batch)
        disc_sum = 0
        for ilevel in range(self.x_by_level):
            disc_sum += self.level_discs[ilevel](
                batch.x_by_level[ilevel], batch.batch_by_level[ilevel]
            )
            # if ilevel == self.x_by_level - 1:
            #     continue
            # disc_sum += self.branching_discs(x_level)

        return disc_sum
