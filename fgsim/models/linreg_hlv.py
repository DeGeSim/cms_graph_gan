import torch
from torch import nn

from ..config import conf

nfeatures = conf.model.dyn_features + conf.model.static_features
nhlv = len(conf.loader.keylist) - 2 - 1


def hlv():
    return nn.Sequential(
        nn.Linear(nhlv, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.dyn_features),
        nn.ReLU(),
    )


def batch_to_hlvs(batch):
    varsL = [
        batch[k] for k in conf.loader.keylist if k not in ["energy", "ECAL", "HCAL"]
    ]
    X = torch.vstack(varsL).float().T
    return X


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.end_lin = nn.Linear(conf.model.dyn_features, 1)
        self.hlv = hlv()

    def forward(self, batch):
        X = self.hlv(batch_to_hlvs(batch))
        x = self.end_lin(X)
        return x
