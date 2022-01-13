import torch
import torch.nn as nn
from torch_geometric.data import Data


class GlobalDeepAggr(nn.Module):
    def __init__(self, pre_nn: nn.Module, post_nn: nn.Module):
        super().__init__()
        self.pre_nn = pre_nn
        self.post_nn = post_nn

    def forward(self, graph: Data):
        ftx_mtx = self.pre_nn(graph.x)
        global_ftx = torch.sum(ftx_mtx, -2)
        global_ftx = self.post_nn(global_ftx)
        return global_ftx
