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
        self.conv = get_conv()
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
            x = self.conv(addstatic(x), batch.edge_index)
            x = self.node_dnn(addstatic(x))
            cuda_clear()

        x = global_add_pool(x, batch.batch, size=batch.num_graphs)

        x = self.hlv_dnn(torch.hstack((hlvs, x)))
        return x
