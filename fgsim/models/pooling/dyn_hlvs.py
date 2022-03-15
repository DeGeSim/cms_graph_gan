import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_mean_pool

from fgsim.models.dnn_gen import dnn_gen


class DynHLVsLayer(nn.Module):
    def __init__(self, n_features, n_global, n_events: int, device: torch.device):
        super().__init__()
        self.n_features = n_features
        self.n_global = n_global
        self.n_events = n_events
        self.pre_nn: nn.Module = dnn_gen(self.n_features, self.n_features).to(
            device
        )
        self.post_nn: nn.Module = dnn_gen(self.n_features * 2, self.n_global).to(
            device
        )

    def forward(self, graph: Data):
        ftx_mtx = self.pre_nn(graph.x)
        gsum = global_add_pool(ftx_mtx, graph.event)
        gmean = global_mean_pool(ftx_mtx, graph.event)
        global_ftx = self.post_nn(torch.hstack([gsum, gmean]))
        return global_ftx
