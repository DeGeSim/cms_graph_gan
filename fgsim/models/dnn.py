import torch
from torch import nn

from fgsim.config import conf

n_dyn = conf.model.dyn_features
n_hlvs = len(conf.loader.hlvs)
n_node_features = len(conf.loader.cell_prop_keys)

n_all_features = n_dyn + n_hlvs + n_node_features


def hlv():
    return nn.Sequential(
        nn.Linear(n_hlvs, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.dyn_features),
        nn.ReLU(),
    )


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.hlv = hlv()
        self.end_lin = nn.Linear(conf.model.dyn_features, 1)

    def forward(self, batch):
        X = batch.hlvs
        X = self.hlv(X)
        X = self.end_lin(X)
        return X
