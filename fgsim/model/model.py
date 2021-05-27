from functools import reduce

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_add_pool

from ..config import conf, device
from ..geo.graph import num_node_features

imgpixels = reduce(lambda a, b: a * b, conf["mapper"]["calo_img_shape"])


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 30)
        self.conv2 = GCNConv(30, 30)
        self.conv3 = GCNConv(30, 30)
        self.lin = Linear(30, 1)

    def forward(self, batch):
        adj_mtx_coo = batch.edge_index
        x = self.conv1(batch.x, adj_mtx_coo)
        x = F.relu(x)

        x = self.conv2(x, adj_mtx_coo)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, adj_mtx_coo)
        x = F.relu(x)

        x = global_add_pool(x, batch.batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        return x
