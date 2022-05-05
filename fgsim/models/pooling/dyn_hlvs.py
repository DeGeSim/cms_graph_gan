import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool

from fgsim.models.ffn import FFN


class DynHLVsLayer(nn.Module):
    def __init__(self, n_features, n_global, batch_size: int, device: torch.device):
        super().__init__()
        self.n_features = n_features
        self.n_global = n_global
        self.batch_size = batch_size
        self.pre_nn: nn.Module = FFN(self.n_features, self.n_features).to(device)
        self.post_nn: nn.Module = FFN(self.n_features, self.n_global).to(device)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        ftx_mtx = self.pre_nn(x)
        gsum = global_add_pool(ftx_mtx, batch)
        global_ftx = self.post_nn(gsum)
        return global_ftx
