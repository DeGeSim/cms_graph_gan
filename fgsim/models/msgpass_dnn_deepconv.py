import torch
import torch.nn.functional as F
from torch import nn

# import torch_geometric
from torch.nn import Linear
from torch_geometric.nn import GINConv, global_add_pool

from ..config import conf, device

nfeatures = conf.model.dyn_features + conf.model.static_features


def dnn():
    return nn.Sequential(
        nn.Linear(nfeatures, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.deeplayer_nodes),
        nn.ReLU(),
        nn.Linear(conf.model.deeplayer_nodes, conf.model.dyn_features),
        nn.ReLU(),
    )


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()

        self.deep_conv = GINConv(dnn(), train_eps=True)
        self.node_dnn = dnn()
        self.lin = Linear(conf.model.dyn_features, 1)

    def forward(self, batch):
        def addstatic(
            x, mask=torch.ones(len(batch.x), dtype=torch.bool, device=device)
        ):
            return torch.hstack((x[mask], batch.feature_mtx_static[mask]))

        x = torch.hstack(
            (
                batch.x,
                torch.zeros(
                    (len(batch.x), conf.model.dyn_features - 1), device=device
                ),
            )
        )

        for _ in range(conf.model.nprop):
            x = self.deep_conv(addstatic(x), batch.edge_index)
            x = self.node_dnn(addstatic(x))

        x = global_add_pool(x, batch.batch)
        x = self.lin(x)
        x = F.relu(x)

        return x
