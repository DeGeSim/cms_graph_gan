import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool

from fgsim.config import conf


class ModelClass(torch.nn.Module):
    def __init__(self, activation):
        super(ModelClass, self).__init__()
        self.activation = getattr(torch.nn, activation)()
        n_features = conf.loader.n_features
        self.hlv_dnn = nn.Sequential(
            nn.Linear(n_features * 2, n_features * 2),
            nn.ReLU(),
            nn.Linear(n_features * 2, n_features * 2),
            nn.ReLU(),
            nn.Linear(n_features * 2, n_features),
            nn.ReLU(),
            nn.Linear(n_features, 1),
            nn.ReLU(),
        )

    def forward(self, batch):
        x = batch.x
        x = torch.hstack(
            [
                global_add_pool(x, batch.batch, size=batch.num_graphs),
                global_mean_pool(x, batch.batch, size=batch.num_graphs),
            ]
        )
        x = self.hlv_dnn(x)
        return self.activation(x)
