import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_add_pool

from ..config import conf
from ..geo.graph import num_node_dyn_features as initial_dyn_features

nfeatures = conf.model.dyn_features + conf.model.static_features


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.upscale_conv = GCNConv(initial_dyn_features, conf.model.dyn_features)
        self.conv1 = GCNConv(nfeatures, conf.model.dyn_features)
        self.conv2 = GCNConv(nfeatures, conf.model.dyn_features)
        self.conv3 = GCNConv(nfeatures, conf.model.dyn_features)
        self.lin = Linear(nfeatures, 1)

    def forward(self, batch):
        def addstatic(x):
            return torch.hstack((x, batch.feature_mtx_static))

        x = self.upscale_conv(batch.x, batch.edge_index)

        x = self.conv1(addstatic(x), batch.edge_index)
        x = F.relu(x)

        x = self.conv2(addstatic(x), batch.edge_index)
        x = F.relu(x)

        x = self.conv3(addstatic(x), batch.edge_index)
        x = F.relu(x)

        x = global_add_pool(x, batch.batch)
        x = self.lin(x)
        return x
