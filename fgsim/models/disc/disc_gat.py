import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, global_add_pool, knn_graph
from torch_geometric.utils import to_dense_batch

from fgsim.models.common.ffn import FFN

# https://proceedings.mlr.press/v80/achlioptas18a.html
# https://github.com/optas/latent_3d_points


class ModelClass(nn.Module):
    def __init__(self, n_features: int, n_points: int, n_cond: int):
        super().__init__()
        self.n_features = n_features
        self.n_points = n_points
        self.n_cond = n_cond

        self.rgan_down_dim = 4
        self.gnn_dim = 5
        self.n_heads = 3

        features = [self.n_features, 16, 32, 64]
        self.fc_layers = nn.ModuleList(
            [
                nn.Conv1d(features[inx], features[inx + 1], kernel_size=1, stride=1)
                for inx in range(len(features) - 1)
            ]
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.rgan_downproj = FFN(features[-1], self.rgan_down_dim, n_layers=2)

        self.lin = FFN(
            self.n_features + self.rgan_down_dim + self.n_cond,
            self.gnn_dim,
            n_layers=2,
        )
        self.gin1 = GINConv(FFN(self.gnn_dim, self.gnn_dim, n_layers=2))
        self.gat = GATv2Conv(self.gnn_dim, self.gnn_dim, heads=self.n_heads)
        self.gin2 = GINConv(
            FFN(self.gnn_dim * self.n_heads, self.gnn_dim, n_layers=2)
        )

        self.final_layer = FFN(self.rgan_down_dim + self.gnn_dim + self.n_cond, 1)

    def forward(self, batch, cond):
        n_features = batch.x.shape[1]
        batch_size = batch.batch[-1] + 1

        edge_index = knn_graph(batch.x, 8, batch.batch, loop=True)
        cond = cond.reshape(batch_size, self.n_cond)
        # rgan part
        feat = (
            to_dense_batch(batch.x)[0]  # pad the feature vector
            .reshape(batch_size, self.n_points, n_features)
            .transpose(1, 2)
        )
        max_hits = feat.size(2)
        for inx in range(len(self.fc_layers)):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
        feat = F.max_pool1d(input=feat, kernel_size=max_hits).squeeze(-1)
        rganout = self.rgan_downproj(feat)

        x = self.lin(
            torch.hstack((batch.x, cond[batch.batch], rganout[batch.batch]))
        )
        edge_index = knn_graph(x, 8, batch.batch, loop=True)
        x = self.gin1(x, edge_index)
        x = self.gat(x, edge_index)
        x = self.gin2(x, edge_index)
        gnnout = global_add_pool(x, batch=batch.batch, size=batch_size)

        out = self.final_layer(torch.hstack((rganout, cond, gnnout))).squeeze()
        assert not torch.any(torch.isnan(out))
        return out
