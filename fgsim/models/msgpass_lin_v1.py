import torch
import torch.nn.functional as F
from torch import nn

# import torch_geometric
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_add_pool

from ..config import conf

nfeatures = conf.model.dyn_features + conf.model.static_features


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.upscale_conv = GCNConv(1, conf.model.dyn_features)
        self.conv1 = GCNConv(nfeatures, conf.model.dyn_features)
        self.conv2 = GCNConv(nfeatures, conf.model.dyn_features)
        self.conv3 = GCNConv(nfeatures, conf.model.dyn_features)
        self.node_dnn = nn.Sequential(
            nn.Linear(nfeatures, nfeatures),
            nn.ReLU(),
            nn.Linear(nfeatures, conf.model.dyn_features),
            nn.ReLU(),
            nn.Linear(conf.model.dyn_features, conf.model.dyn_features),
            nn.ReLU(),
        )
        self.lin = Linear(conf.model.dyn_features, 1)

    def forward(self, batch):
        def addstatic(feature_mtx):
            return torch.hstack((feature_mtx, batch.feature_mtx_static))

        x = self.upscale_conv(batch.x, batch.edge_index)

        x = self.conv1(addstatic(x), batch.edge_index)
        x = F.relu(x)

        x = self.node_dnn(addstatic(x))
        x = F.relu(x)

        x = self.conv2(addstatic(x), batch.edge_index)
        x = F.relu(x)

        x = self.node_dnn(addstatic(x))
        x = F.relu(x)

        x = self.conv3(addstatic(x), batch.edge_index)
        x = F.relu(x)

        x = self.node_dnn(addstatic(x))
        x = F.relu(x)

        x = global_add_pool(x, batch.batch)
        x = self.lin(x)
        x = F.relu(x)

        return x
