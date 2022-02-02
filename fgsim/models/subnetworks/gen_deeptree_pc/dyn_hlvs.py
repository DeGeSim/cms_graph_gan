import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool


class DynHLVsLayer(nn.Module):
    def __init__(self, pre_nn: nn.Module, post_nn: nn.Module, n_events: int):
        super().__init__()
        self.pre_nn = pre_nn
        self.post_nn = post_nn
        self.n_events = n_events

    def forward(self, graph: Data):
        ftx_mtx = self.pre_nn(graph.x)
        gsum = global_add_pool(ftx_mtx, graph.event)
        gmean = global_mean_pool(ftx_mtx, graph.event)
        global_ftx = self.post_nn(torch.hstack([gsum, gmean]))
        return global_ftx
