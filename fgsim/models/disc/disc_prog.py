import torch

from fgsim.config import conf

# from .disc_branching import ModelClass as BranchingDisc
from .disc_graphgym import ModelClass as LevelDisc


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.branches = conf.tree.branches
        self.features = conf.tree.features
        self.levels = len(conf.tree.features)

        self.level_discs = torch.nn.ModuleList(
            [LevelDisc() for _ in range(self.levels)]
        )
        # self.branching_discs = torch.nn.ModuleList(
        #     [BranchingDisc() for _ in range(self.levels - 1)]
        # )

    def forward(self, batch):
        disc_sum = 0
        for ilevel in range(self.levels):
            x_level = batch.levels[0]
            disc_sum += self.level_discs(x_level)
            # if ilevel == self.levels - 1:
            #     continue
            # disc_sum += self.branching_discs(x_level)

        return disc_sum
