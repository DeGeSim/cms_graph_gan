import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_add_pool

from ..config import conf
from ..geo.graph import num_node_dyn_features as initial_dyn_features

nfeatures = conf.model.dyn_features + conf.model.static_features


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.upscale_conv = GCNConv(initial_dyn_features, conf.model.dyn_features)
        self.inlayer_conv = GCNConv(nfeatures, conf.model.dyn_features)
        self.forward_conv = GCNConv(nfeatures, conf.model.dyn_features)
        self.backward_conv = GCNConv(nfeatures, conf.model.dyn_features)
        self.lin = Linear(conf.model.dyn_features, 1)

    def forward(self, batch):
        x = self.upscale_conv(batch.x, batch.edge_index)

        # forward_edges_per_layer[i] map i->i+1 last layer empty
        # backward_edges_per_layer[i] map i->i-1 first first empty
        def addstatic(x, mask=torch.ones(len(batch.x))):
            return torch.hstack((x[mask], batch.feature_mtx_static[mask]))

        for _ in range(conf.model.nprop):

            for ilayer in range(conf.nlayers):
                # forwards
                # the last time is just inlayer MPL
                inner_inp_mask = batch.mask_inp_innerL[ilayer]
                inner_outp_mask = batch.mask_outp_innerL[ilayer]

                partial_inner = self.inlayer_conv(
                    addstatic(x, inner_inp_mask), batch.inner_edges_per_layer[ilayer]
                )
                x[batch.layers == ilayer] = partial_inner[inner_outp_mask]

                if ilayer == conf.nlayers - 1:
                    continue
                forward_inp_mask = batch.mask_inp_forwardL[ilayer]
                forward_outp_mask = batch.mask_outp_forwardL[ilayer]
                partial_forward = self.forward_conv(
                    addstatic(x, forward_inp_mask),
                    batch.forward_edges_per_layer[ilayer],
                )
                x[batch.layers == ilayer + 1] = partial_forward[forward_outp_mask]

            x = F.relu(x)

            # backward
            # ilayer goes from nlayers - 1 to nlayers - 2 to ... 1
            for ilayer in range(conf.nlayers - 1, 0, -1):
                # backward
                # the last time is just inlayer MPL
                backward_inp_mask = batch.mask_inp_backwardL[ilayer]
                backward_outp_mask = batch.mask_outp_backwardL[ilayer]
                partial_backward = self.backward_conv(
                    addstatic(x, backward_inp_mask),
                    batch.backward_edges_per_layer[ilayer],
                )
                x[batch.layers == ilayer - 1] = partial_backward[backward_outp_mask]

                inner_inp_mask = batch.mask_inp_innerL[ilayer - 1]
                inner_outp_mask = batch.mask_outp_innerL[ilayer - 1]
                partial_inner = self.inlayer_conv(
                    addstatic(x, inner_inp_mask),
                    batch.inner_edges_per_layer[ilayer - 1],
                )
                x[batch.layers == ilayer - 1] = partial_inner[inner_outp_mask]

            x = F.relu(x)

        x = global_add_pool(x, batch.batch)
        x = self.lin(x)
        feature_mtx = F.relu(feature_mtx)
        return x
