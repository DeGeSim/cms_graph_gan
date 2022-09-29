import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
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
        self.gnn_dim = 30
        self.n_heads = 3
        self.k = 30

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
            # n_layers=2,
        )
        self.gin1 = pygnn.GINConv(FFN(self.gnn_dim, self.gnn_dim))
        self.mpdown1 = FFN(self.gnn_dim + self.gnn_dim, self.gnn_dim)
        # self.gat = pygnn.GATv2Conv(self.gnn_dim, self.gnn_dim, heads=self.n_heads)
        self.gin2 = pygnn.GINConv(FFN(self.gnn_dim, self.gnn_dim))
        self.mpdown2 = FFN(self.gnn_dim + self.gnn_dim, self.gnn_dim)

        self.final_layer = FFN(
            self.rgan_down_dim + self.gnn_dim + self.n_cond, 1, final_linear=False
        )

    def forward(self, batch, cond):
        n_features = batch.x.shape[1]
        batch_size = batch.batch[-1] + 1

        cond = cond.reshape(batch_size, self.n_cond)
        # rgan part
        feat = (
            to_dense_batch(
                batch.x, max_num_nodes=self.n_points, batch_size=batch_size
            )[
                0
            ]  # pad the feature vector
            .reshape(batch_size, self.n_points, n_features)
            .transpose(1, 2)
        )
        max_hits = feat.size(2)
        for inx in range(len(self.fc_layers)):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)
        feat = F.max_pool1d(input=feat, kernel_size=max_hits).squeeze(-1)
        rganout = self.rgan_downproj(feat)

        # gnn part
        x = self.lin(
            torch.hstack((batch.x, cond[batch.batch], rganout[batch.batch]))
        )
        edge_index = pygnn.knn_graph(x, 8, batch.batch, loop=True)
        mpout = self.gin1(x, edge_index)
        mpaggr = pygnn.global_add_pool(mpout, batch=batch.batch, size=batch_size)
        x = self.mpdown1(torch.hstack((x, mpaggr[batch.batch])))

        # x = self.gat(x, edge_index)
        mpout = self.gin2(x, edge_index)
        mpaggr = pygnn.global_add_pool(mpout, batch=batch.batch, size=batch_size)
        x = self.mpdown2(torch.hstack((x, mpaggr[batch.batch])))
        ggnout = pygnn.global_add_pool(x, batch=batch.batch, size=batch_size)

        out = self.final_layer(torch.hstack((rganout, cond, ggnout))).squeeze()
        assert not torch.any(torch.isnan(out))
        return out
