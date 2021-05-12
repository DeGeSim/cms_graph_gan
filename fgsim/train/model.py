from functools import reduce

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from ..config import conf, device
from ..geo.graph import num_node_features

imgpixels = reduce(lambda a, b: a * b, conf["mapper"]["calo_img_shape"])


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, 20)
        self.conv2 = GCNConv(20, num_node_features)
        self.conv3 = GCNConv(num_node_features, 1)

    def forward(self, batch):
        resL = []
        for graph in batch:
            feature_mtx = graph["feature_mtx"]
            adj_mtx_coo = graph["adj_mtx_coo"]

            feature_mtx = torch.tensor(feature_mtx, dtype=torch.float32, device=device)
            adj_mtx_coo = torch.tensor(adj_mtx_coo, dtype=torch.int64, device=device)
            adj_mtx_coo = adj_mtx_coo.T

            x = self.conv1(feature_mtx, adj_mtx_coo)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, adj_mtx_coo)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, adj_mtx_coo)
            resL.append(torch.sum(x))
        return torch.stack(resL)