import torch
from torch.nn import LeakyReLU, Linear, PReLU
from torch_geometric.nn import GeneralConv, Sequential, global_add_pool, knn_graph

from fgsim.config import conf, device


class ModelClass(torch.nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        n_features = conf.loader.n_features
        self.hlv_dnn = torch.nn.Sequential(
            Linear(n_features, n_features * 4),
            LeakyReLU(0.2),
            Linear(n_features * 4, n_features * 4),
            LeakyReLU(0.2),
            Linear(n_features * 4, n_features),
            LeakyReLU(0.2),
            Linear(n_features, 1),
        ).to(device)
        self.graph_gym_nn = Sequential(
            "x, edge_index",
            [
                (Linear(n_features, n_features), "x -> x"),
                (PReLU(n_features), "x -> x"),
                (Linear(n_features, n_features), "x -> x"),
                (PReLU(n_features), "x -> x"),
                (GeneralConv(n_features, n_features), "x, edge_index -> x"),
                (GeneralConv(n_features, n_features), "x, edge_index -> x"),
                (GeneralConv(n_features, n_features), "x, edge_index -> x"),
                (GeneralConv(n_features, n_features), "x, edge_index -> x"),
                (Linear(n_features, n_features), "x -> x"),
                (PReLU(n_features), "x -> x"),
                (Linear(n_features, n_features), "x -> x"),
                (PReLU(n_features), "x -> x"),
            ],
        ).to(device)

    def forward(self, batch):
        batch.edge_index = knn_graph(x=batch.x, k=6, batch=batch.batch)
        x_graph = self.graph_gym_nn(batch.x, batch.edge_index)
        x_aggr = global_add_pool(x_graph, batch.batch, size=batch.num_graphs)
        x = self.hlv_dnn(x_aggr)
        return x
