import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

from ..config import conf, device
from ..utils.cuda_clear import cuda_clear

n_node_features = conf.model.dyn_features + conf.model.static_features
n_hl_features = len(conf.loader.keylist) - 2 - 1
n_all_features = n_node_features + n_hl_features


def get_hlv_dnn():
    return nn.Sequential(
        nn.Linear(
            n_hl_features + conf.model.dyn_features, conf.model.deeplayer_nodes
        ),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, 1),
        nn.ReLU(),
    )


def get_node_dnn():
    return nn.Sequential(
        nn.Linear(n_all_features, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.dyn_features),
        nn.ReLU(),
    )


def get_conv():
    conv_dnn = get_node_dnn()
    return GINConv(conv_dnn, train_eps=True)


def batch_to_hlvs(batch):
    varsL = [
        batch[k] for k in conf.loader.keylist if k not in ["energy", "ECAL", "HCAL"]
    ]
    X = torch.vstack(varsL).float().T
    return X


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.inlayer_conv = get_conv()
        self.forward_conv = get_conv()
        self.backward_conv = get_conv()
        self.node_dnn = get_node_dnn()
        self.hlv_dnn = get_hlv_dnn()

    def forward(self, batch):
        hlvs = batch_to_hlvs(batch)

        def addstatic(
            x, mask=torch.ones(len(batch.x), dtype=torch.bool, device=device)
        ):
            return torch.hstack(
                (x[mask], batch.feature_mtx_static[mask], hlvs[batch.batch[mask]])
            )

        x = torch.hstack(
            (
                batch.x,
                torch.zeros(
                    (len(batch.x), conf.model.dyn_features - 1), device=device
                ),
            )
        )
        for _ in range(conf.model.nprop):
            # forwards
            # the last time is just inlayer MPL
            for ilayer in range(conf.nlayers):

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
                x[targetlayermask] = self.node_dnn(addstatic(x, targetlayermask))
                cuda_clear()

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

                inner_inp_mask = batch.mask_inp_innerL[ilayer - 1]
                inner_outp_mask = batch.mask_outp_innerL[ilayer - 1]
                partial_inner = self.inlayer_conv(
                    addstatic(x, inner_inp_mask),
                    batch.inner_edges_per_layer[ilayer - 1],
                )
                x[targetlayermask] = partial_inner[inner_outp_mask]
                del partial_inner, inner_inp_mask, inner_outp_mask

                x[targetlayermask] = self.node_dnn(addstatic(x, targetlayermask))
                cuda_clear()

        x = global_add_pool(x, batch.batch, size=batch.num_graphs)

        x = self.hlv_dnn(torch.hstack((hlvs, x)))
        return x
