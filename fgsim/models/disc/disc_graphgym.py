import torch
from torch.nn import Linear, PReLU
from torch_geometric.nn import (
    BatchNorm,
    GeneralConv,
    JumpingKnowledge,
    Sequential,
    global_add_pool,
    knn_graph,
)

from fgsim.config import conf, device
from fgsim.models.dnn_gen import dnn_gen


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        n_features = conf.loader.n_features

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
        ).to(device)
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
        ).to(device)

        self.hlv_dnn = dnn_gen(n_features * (len(self.convs) + 1), 1).to(device)

    def forward(self, data):
        data.edge_index = knn_graph(x=data.x, k=6, batch=data.batch)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.pre_nn(x)
        xs = [x]
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
            xs += [x]
        x = self.jk(xs)
        x_aggr = global_add_pool(x, batch, size=data.num_graphs)
        x = self.hlv_dnn(x_aggr)
        return x
