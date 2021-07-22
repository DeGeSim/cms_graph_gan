import torch
from torch import nn
from torch_geometric.nn import global_add_pool

from ..config import conf

nfeatures = conf.model.dyn_features + conf.model.static_features


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.end_lin = nn.Linear(1, 1)

    def forward(self, batch):
        x = global_add_pool(batch.x, batch.batch)
        x = self.end_lin(x)
        return x
