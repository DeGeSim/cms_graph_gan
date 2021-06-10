import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_add_pool

from ..config import conf
from ..geo.graph import num_node_dyn_features as initial_dyn_features

nfeatures = conf.model.dyn_features + conf.model.static_features


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.upscale_conv = GCNConv(initial_dyn_features, conf.model.dyn_features)
        self.inlayer_conv = GCNConv(nfeatures, conf.model.dyn_features)
        self.forward_conv = GCNConv(nfeatures, conf.model.dyn_features)
        self.backward_conv = GCNConv(nfeatures, conf.model.dyn_features)
        self.lin = Linear(conf.model.dyn_features, 1)

    def forward(self, batch):
        x = self.upscale_conv(batch.x, batch.edge_index)

        # forward_edges_per_layer[i] map i->i+1 last layer empty
        # backward_edges_per_layer[i] map i->i-1 first first empty
        def addstatic(x):
            return torch.hstack((x, batch.feature_mtx_static))

        for _ in range(conf.model.nprop):

            for ilayer in range(conf.nlayers):
                # forwards
                # the last time is just inlayer MPL
                layermask = batch.layers == ilayer

                partial_inner = self.inlayer_conv(
                    addstatic(x), batch.inner_edges_per_layer[ilayer]
                )
                x[layermask] = partial_inner[layermask]

                next_layer_mask = batch.layers == (ilayer + 1)
                partial_forward = self.forward_conv(
                    addstatic(x), batch.forward_edges_per_layer[ilayer]
                )
                x[next_layer_mask] = partial_forward[next_layer_mask]

            x = F.relu(x)

            # backwards
            for ilayer in range(conf.nlayers - 1, -1, -1):
                # backwards
                # the last time is just inlayer MPL
                previous_layer_mask = batch.layers == (ilayer - 1)
                partial_forward = self.forward_conv(
                    addstatic(x), batch.backward_edges_per_layer[ilayer]
                )
                x[previous_layer_mask] = partial_inner[previous_layer_mask]

                # in layer
                layermask = batch.layers == ilayer
                partial_backward = self.inlayer_conv(
                    addstatic(x), batch.inner_edges_per_layer[ilayer]
                )
                x[layermask] = partial_backward[layermask]

            x = F.relu(x)

        x = global_add_pool(x, batch.batch)
        x = self.lin(x)
        return x
