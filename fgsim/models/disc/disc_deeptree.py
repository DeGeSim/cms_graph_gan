import torch
from torch import nn
from torch_geometric.nn import global_add_pool, knn_graph
from torch_geometric.nn.pool import TopKPooling

from fgsim.models.common import FFN, MPLSeq

ffn_param = {"bias": False, "n_layers": 2, "hidden_layer_size": 40, "dropout": 0.3}


class TSumTDisc(nn.Module):
    """Classifies PC via FNN -> Add -> FNN"""

    def __init__(self, n_ftx_out, n_ftx_disc) -> None:
        super().__init__()
        self.disc_emb = EPiC(
            n_in=n_ftx_out, n_latent=5, n_global=4, n_out=n_ftx_disc
        )
        self.disc = FFN(n_ftx_disc, 1, **ffn_param, final_linear=True)

    def forward(self, x, batch):
        x = self.disc_emb(x, batch)
        x = global_add_pool(x, batch)
        x = self.disc(x)
        return x


class EPiC(nn.Module):
    """update with global vector"""

    def __init__(self, n_in, n_latent, n_global, n_out) -> None:
        super().__init__()
        self.emb_nn = FFN(n_in, n_latent, **ffn_param)
        self.global_nn = FFN(n_latent, n_global, **ffn_param)
        self.out_nn = FFN(n_latent + n_global, n_out, **ffn_param)

    def forward(self, x, batch):
        x = self.emb_nn(x)
        x_aggr = global_add_pool(x, batch)
        x_global = self.global_nn(x_aggr)
        x = self.out_nn(torch.hstack([x, x_global[batch]]))
        return x


class ModelClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes = [30, 6, 2, 1]
        self.features = [3, 6, 12, 18]
        self.n_levels = len(self.nodes)
        self.n_ftx_space = 2
        self.n_ftx_disc = 5
        self.n_ftx_latent = 10

        self.embeddings = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.pcdiscs = nn.ModuleList()

        for ilevel in range(self.n_levels - 1):
            self.embeddings.append(
                # FFN(3, 1, **ffn_param)
                LevelLayer(
                    n_ftx_in=self.features[ilevel],
                    n_ftx_out=self.features[ilevel + 1],
                    n_ftx_space=self.n_ftx_space,
                    n_ftx_latent=self.n_ftx_latent,
                    n_ftx_disc=self.n_ftx_disc,
                )
            )
            self.pools.append(
                TopKPooling(
                    in_channels=1,  # self.features[ilevel + 1],
                    ratio=self.nodes[ilevel + 1],
                )
            )
            self.pcdiscs.append(TSumTDisc(self.features[ilevel], self.n_ftx_disc))
        self.last_level_disc = TSumTDisc(self.features[-1], self.n_ftx_disc)

    def forward(self, batch, condition):
        x: torch.Tensor
        x, batchidx = batch.x, batch.batch
        x_disc = torch.zeros((batch.num_graphs, 1), dtype=x.dtype, device=x.device)
        for ilevel in range(self.n_levels - 1):
            x_disc += self.pcdiscs[ilevel](x, batchidx)

            x = self.embeddings[ilevel](x, batchidx, condition)[1]
            _, _, _, batchidx, perm, _ = self.pools[ilevel](
                x=x,
                edge_index=torch.empty(2, 0, dtype=torch.long, device=x.device),
                batch=batchidx,
            )
            x = x[perm]

        x_disc += self.last_level_disc(x, batchidx)
        return x_disc


def signsqrt(x: torch.Tensor):
    return torch.sign(x) * torch.sqrt(torch.abs(x))


def skipadd(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[:-1] == b.shape[:-1]
    ldim = min(a.shape[-1], b.shape[-1])
    a[..., :ldim] += b[..., :ldim]
    return a


class LevelLayer(nn.Module):
    def __init__(
        self, n_ftx_in, n_ftx_out, n_ftx_space, n_ftx_latent, n_ftx_disc
    ) -> None:
        super().__init__()
        self.n_ftx_in = n_ftx_in
        self.n_ftx_space = n_ftx_space
        self.n_ftx_latent = n_ftx_latent
        self.n_ftx_out = n_ftx_out
        self.n_ftx_disc = n_ftx_disc

        self.space_emb = FFN(n_ftx_in, n_ftx_latent, **ffn_param)
        self.mpls = MPLSeq(
            "GINConv",
            self.n_ftx_latent,
            self.n_ftx_latent,
            skip_connecton=True,
            n_hidden_nodes=self.n_ftx_latent,
            layer_param={},
            n_global=0,
            n_cond=0,
            n_mpl=2,
        )
        self.out_emb = FFN(self.n_ftx_latent, self.n_ftx_out, **ffn_param)

    def forward(self, x, batch, condition):
        x = self.space_emb(x)
        ei = knn_graph(x[..., : self.n_ftx_space], batch=batch, k=15)
        x = self.mpls(x=x, edge_index=ei, batch=batch, cond=condition)
        x_emb = self.out_emb(x)
        return x, x_emb  # , ei
