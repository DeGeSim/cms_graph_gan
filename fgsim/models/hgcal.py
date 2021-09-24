import torch
from torch import nn
from torch_geometric.nn import GINConv, global_add_pool

from fgsim.config import conf, device
from fgsim.utils.cuda_clear import cuda_clear

n_dyn = conf.model.dyn_features
n_hlvs = len(conf.loader.hlvs)
n_node_features = len(conf.loader.cell_prop_keys)

n_all_features = n_dyn + n_hlvs + n_node_features


def get_hlv_dnn():
    return nn.Sequential(
        nn.Linear(n_hlvs + n_dyn, conf.model.deeplayer_nodes),
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


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.conv = get_conv()
        self.node_dnn = get_node_dnn()
        self.hlv_dnn = get_hlv_dnn()

    def forward(self, batch):
        def addstatic(
            x, mask=torch.ones(len(batch.x), dtype=torch.bool, device=device)
        ):
            return torch.hstack(
                (
                    x[mask],
                    batch.feature_mtx_static[mask],
                    batch.hlvs[batch.batch[mask]],
                )
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

        x = self.hlv_dnn(torch.hstack((batch.hlvs, x)))
        return x
