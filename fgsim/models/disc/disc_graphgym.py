import torch
from torch.nn import Linear, PReLU
from torch_geometric.data import Data
from torch_geometric.nn import (
    BatchNorm,
    GeneralConv,
    JumpingKnowledge,
    Sequential,
    global_add_pool,
)

from fgsim.models.common import FFN


class ModelClass(torch.nn.Module):
    def __init__(self, n_features, n_nn):
        super(ModelClass, self).__init__()
        self.n_nn = n_nn

        self.jk = JumpingKnowledge(mode="cat")
        self.act = PReLU(n_features)
        self.pre_nn = Sequential(
            "x",
            [
                (Linear(n_features, n_features), "x -> x"),
                (PReLU(n_features), "x -> x"),
                (Linear(n_features, n_features), "x -> x"),
                (PReLU(n_features), "x -> x"),
                (BatchNorm(n_features), "x -> x"),
            ],
        )
        self.convs = torch.nn.ModuleList(
            [GeneralConv(n_features, n_features) for _ in range(4)]
        )

        self.post_nn = Sequential(
            "x",
            [
                (Linear(n_features, n_features), "x -> x"),
                (PReLU(n_features), "x -> x"),
                (Linear(n_features, n_features), "x -> x"),
                (PReLU(n_features), "x -> x"),
            ],
        )

        self.hlv_dnn = FFN(n_features * (len(self.convs) + 1), 1)

    def forward(self, batch: Data):
        x, edge_index, batchidxs = batch.x, batch.edge_index, batch.batch

        x = self.pre_nn(x)
        xs = [x]
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
            xs += [x]
        x = self.jk(xs)
        x_aggr = global_add_pool(x, batchidxs)
        x = self.hlv_dnn(x_aggr)
        return x
