import torch
from torch import nn

from fgsim.config import conf

nfeatures = conf.model.dyn_features + conf.model.static_features


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.end_lin = nn.Linear(2, 1)

    def forward(self, batch):
        X = torch.vstack([batch["ECAL_E"], batch["HCAL_E"]]).float().T
        X = self.end_lin(X)
        return X
