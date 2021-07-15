import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_add_pool

from ..config import conf
from ..geo.graph import num_node_dyn_features as initial_dyn_features
from ..utils.cuda_clear import cuda_clear

nfeatures = conf.model.dyn_features + conf.model.static_features


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.upscale_conv = GCNConv(initial_dyn_features, conf.model.dyn_features)
        self.inlayer_conv = GCNConv(nfeatures, conf.model.dyn_features)
        self.forward_conv = GCNConv(nfeatures, conf.model.dyn_features)
        self.backward_conv = GCNConv(nfeatures, conf.model.dyn_features)
        self.node_dnn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(nfeatures, nfeatures),
            nn.ReLU(),
            nn.Linear(nfeatures, nfeatures),
            nn.ReLU(),
            nn.Linear(nfeatures, conf.model.dyn_features),
            nn.ReLU(),
        )
        self.end_lin = nn.Linear(conf.model.dyn_features, 1)

    def forward(self, batch):
        x = self.upscale_conv(batch.x, batch.edge_index)

        # forward_edges_per_layer[i] map i->i+1 last layer empty
        # backward_edges_per_layer[i] map i->i-1 first first empty
        def addstatic(x, mask=torch.ones(len(batch.x), dtype=torch.bool)):
            return torch.hstack((x[mask], batch.feature_mtx_static[mask]))

        for _ in range(conf.model.nprop):

            for ilayer in range(conf.nlayers):
                # forwards
                # the last time is just inlayer MPL
                inner_inp_mask = batch.mask_inp_innerL[ilayer]
                inner_outp_mask = batch.mask_outp_innerL[ilayer]

                sourcelayermask = batch.layers == ilayer
                targetlayermask = batch.layers == ilayer + 1

                partial_inner = self.inlayer_conv(
                    addstatic(x, inner_inp_mask),
                    batch.inner_edges_per_layer[ilayer],
                )
                x[sourcelayermask] = partial_inner[inner_outp_mask]
                del partial_inner, inner_inp_mask, inner_outp_mask
                cuda_clear()

                if ilayer == conf.nlayers - 1:
                    continue
                forward_inp_mask = batch.mask_inp_forwardL[ilayer]
                forward_outp_mask = batch.mask_outp_forwardL[ilayer]
                partial_forward = self.forward_conv(
                    addstatic(x, forward_inp_mask),
                    batch.forward_edges_per_layer[ilayer],
                )
                x[targetlayermask] = partial_forward[forward_outp_mask]
                del partial_forward, forward_inp_mask, forward_outp_mask
                cuda_clear()
                x[targetlayermask] = self.node_dnn(addstatic(x, targetlayermask))

            # backward
            # ilayer goes from nlayers - 1 to nlayers - 2 to ... 1
            for ilayer in range(conf.nlayers - 1, 0, -1):
                # backward
                # the last time is just inlayer MPL
                backward_inp_mask = batch.mask_inp_backwardL[ilayer]
                backward_outp_mask = batch.mask_outp_backwardL[ilayer]

                sourcelayermask = batch.layers == ilayer - 2
                targetlayermask = batch.layers == ilayer - 1

                partial_backward = self.backward_conv(
                    addstatic(x, backward_inp_mask),
                    batch.backward_edges_per_layer[ilayer],
                )
                x[targetlayermask] = partial_backward[backward_outp_mask]
                del partial_backward, backward_inp_mask, backward_outp_mask
                cuda_clear()

                inner_inp_mask = batch.mask_inp_innerL[ilayer - 1]
                inner_outp_mask = batch.mask_outp_innerL[ilayer - 1]
                partial_inner = self.inlayer_conv(
                    addstatic(x, inner_inp_mask),
                    batch.inner_edges_per_layer[ilayer - 1],
                )
                x[targetlayermask] = partial_inner[inner_outp_mask]
                del partial_inner, inner_inp_mask, inner_outp_mask
                cuda_clear()
                x[targetlayermask] = self.node_dnn(addstatic(x, targetlayermask))

        x = global_add_pool(x, batch.batch)
        x = self.end_lin(x)
        x = nn.functional.relu(x)
        return x
