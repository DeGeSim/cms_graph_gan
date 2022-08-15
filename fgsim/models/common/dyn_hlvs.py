import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from .ffn import FFN


class DynHLVsLayer(nn.Module):
    def __init__(self, n_features, n_global, batch_size: int, **kwargs):
        super().__init__()
        self.n_features = n_features
        self.n_global = n_global
        self.batch_size = batch_size
        self.pre_nn: nn.Module = FFN(self.n_features, self.n_features)
        self.post_nn: nn.Module = FFN(self.n_features, self.n_global)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        ftx_mtx = self.pre_nn(x)
        gsum = global_add_pool(ftx_mtx, batch)
        global_ftx = self.post_nn(gsum)
        return global_ftx.reshape(-1, self.n_global)
