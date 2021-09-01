import torch
from torch import nn
from torch_geometric.nn import GCNConv, GINConv, global_add_pool

from fgsim.config import conf, device
from fgsim.utils.cuda_clear import cuda_clear

nfeatures = conf.model.dyn_features + conf.model.static_features


def getconv():
    conv_dnn = nn.Sequential(
        nn.Linear(nfeatures, nfeatures),
        nn.ReLU(),
        nn.Linear(nfeatures, conf.model.dyn_features),
        nn.ReLU(),
        nn.Linear(conf.model.dyn_features, conf.model.dyn_features),
        nn.ReLU(),
    )
    return GINConv(conv_dnn, train_eps=True)


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.upscale_conv = GCNConv(1, conf.model.dyn_features)
        self.inlayer_conv = getconv()
        self.forward_conv = getconv()
        self.backward_conv = getconv()
        self.node_dnn = nn.Sequential(
            nn.Linear(nfeatures, conf.model.dyn_features),
            nn.ReLU(),
            nn.Linear(conf.model.dyn_features, conf.model.dyn_features),
            nn.ReLU(),
            nn.Linear(conf.model.dyn_features, conf.model.dyn_features),
            nn.ReLU(),
        )
        self.lin = nn.Linear(conf.model.dyn_features, 1)

    def forward(self, batch):
        x = self.upscale_conv(batch.x, batch.edge_index)

        # forward_edges_per_layer[i] map i->i+1 last layer empty
        # backward_edges_per_layer[i] map i->i-1 first first empty
        def addstatic(
            x, mask=torch.ones(len(batch.x), dtype=torch.bool, device=device)
        ):
            return torch.hstack((x[mask], batch.feature_mtx_static[mask]))

        for _ in range(conf.model.nprop):

            for ilayer in range(conf.nlayers):
                # forwards
                # the last time is just inlayer MPL
                inner_inp_mask = batch.mask_inp_innerL[ilayer]
                inner_outp_mask = batch.mask_outp_innerL[ilayer]

                partial_inner = self.inlayer_conv(
                    addstatic(x, inner_inp_mask),
                    batch.inner_edges_per_layer[ilayer],
                )
                x[batch.layers == ilayer] = partial_inner[inner_outp_mask]
                del partial_inner, inner_inp_mask, inner_outp_mask

                if ilayer == conf.nlayers - 1:
                    continue
                forward_inp_mask = batch.mask_inp_forwardL[ilayer]
                forward_outp_mask = batch.mask_outp_forwardL[ilayer]
                partial_forward = self.forward_conv(
                    addstatic(x, forward_inp_mask),
                    batch.forward_edges_per_layer[ilayer],
                )
                x[batch.layers == ilayer + 1] = partial_forward[forward_outp_mask]
                del partial_forward, forward_inp_mask, forward_outp_mask
                cuda_clear()

            x = nn.functional.relu(x)

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
                del partial_backward, backward_inp_mask, backward_outp_mask

                inner_inp_mask = batch.mask_inp_innerL[ilayer - 1]
                inner_outp_mask = batch.mask_outp_innerL[ilayer - 1]
                partial_inner = self.inlayer_conv(
                    addstatic(x, inner_inp_mask),
                    batch.inner_edges_per_layer[ilayer - 1],
                )
                x[batch.layers == ilayer - 1] = partial_inner[inner_outp_mask]
                del partial_inner, inner_inp_mask, inner_outp_mask
                cuda_clear()

            x = nn.functional.relu(x)

            # DNN on the feature matrix
            x = self.node_dnn(addstatic(x))

        x = global_add_pool(x, batch.batch, size=batch.num_graphs)
        x = self.lin(x)
        x = nn.functional.relu(x)
        return x
