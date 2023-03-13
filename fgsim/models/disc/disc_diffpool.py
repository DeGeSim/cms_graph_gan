from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, dense_diff_pool, knn_graph
from torch_geometric.utils import to_dense_adj, to_dense_batch

# from https://colab.research.google.com/github/AntonioLonga/PytorchGeometricTutorial/blob/main/Tutorial16/Tutorial16.ipynb


class GNN(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, normalize=False, lin=True
    ):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        kwgcn = dict(
            improved=normalize,
            bias=False,
        )
        self.convs.append(DenseGCNConv(in_channels, hidden_channels, **kwgcn))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(DenseGCNConv(hidden_channels, hidden_channels, **kwgcn))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(DenseGCNConv(hidden_channels, out_channels, **kwgcn))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj):
        for step in range(len(self.convs)):
            x = self.convs[step](x, adj)
            x = F.relu(x)
            x = self.bns[step](x.transpose(1, 2)).transpose(1, 2)

        return x


class ModelClass(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_features = 3
        max_nodes = 30
        self.num_nodes = [
            -1,
            ceil(0.66 * max_nodes),
            ceil(0.20 * max_nodes),
            1,
        ]
        self.gnn1_pool = GNN(num_features, 64, self.num_nodes[1])
        self.gnn1_embed = GNN(num_features, 64, 64)

        self.gnn2_pool = GNN(64, 64, self.num_nodes[2])
        self.gnn2_embed = GNN(64, 64, 64, lin=False)

        self.gnn3_embed = GNN(64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(64, 64, bias=False)
        self.lin2 = torch.nn.Linear(64, 1, bias=False)

    def forward(self, batch, condition, mask=None):
        x, batchidx = batch.x, batch.batch
        edge_index = knn_graph(x, batch=batchidx, k=5)
        adj = to_dense_adj(edge_index, batch=batchidx)
        x, mask = to_dense_batch(x, batchidx)

        # 1st lvl
        s = self.gnn1_pool(x, adj)
        x = self.gnn1_embed(x, adj)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        assert x.shape == (batch.num_graphs, self.num_nodes[1], 64)
        # edge_index = dense_to_sparse(adj)[0]
        batchidx = torch.arange(
            0, self.num_nodes[1], dtype=torch.long, device=x.device
        ).repeat_interleave(batch.num_graphs)

        # 2nd level
        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s, None)
        assert x.shape == (batch.num_graphs, self.num_nodes[2], 64)
        x = x[0]
        # edge_index = dense_to_sparse(adj)[0]
        batchidx = torch.arange(
            0, self.num_nodes[2], dtype=torch.long, device=x.device
        ).repeat_interleave(batch.num_graphs)

        x = self.gnn3_embed(x, adj)
        x = x.mean(1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
