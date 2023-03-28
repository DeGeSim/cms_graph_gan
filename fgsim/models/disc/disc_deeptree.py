# noqa: F401
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import conv, global_add_pool, knn_graph
from torch_geometric.nn.pool import SAGPooling, TopKPooling, global_max_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from fgsim.config import conf
from fgsim.models.common import FFN, MPLSeq

ffn_param = {
    "bias": False,
    "n_layers": 2,
    "hidden_layer_size": 40,
    "dropout": 0.0,
    "norm": "spectral",
}


class ModelClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes = [30, 6, 1]
        self.features = [3, 20, 40]
        self.n_levels = len(self.nodes)
        self.n_ftx_space = 2
        self.n_ftx_disc = 5
        self.n_ftx_latent = 10

        self.embeddings = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.pcdiscs = nn.ModuleList()

        for ilevel in range(self.n_levels - 1):
            self.embeddings.append(
                Embedding(
                    n_ftx_in=self.features[ilevel],
                    n_ftx_out=self.features[ilevel + 1],
                    n_ftx_space=self.n_ftx_space,
                    n_ftx_latent=self.n_ftx_latent,
                )
            )
            self.pools.append(
                BipartPool(
                    in_channels=self.features[ilevel + 1],
                    ratio=self.nodes[ilevel + 1],
                )
            )
        for ilevel in range(self.n_levels):
            self.pcdiscs.append(TSumTDisc(self.features[ilevel]))

    def forward(self, batch, condition):
        x: torch.Tensor
        x, batchidx = batch.x, batch.batch
        x_disc = torch.zeros((batch.num_graphs, 1), dtype=x.dtype, device=x.device)

        x_lat_list = []

        for ilevel in range(self.n_levels):
            # aggregate latent space features
            x_lat_list.append(global_add_pool(x, batchidx))
            x_lat_list.append(global_max_pool(x, batchidx))

            x_disc += self.pcdiscs[ilevel](x, batchidx)

            if ilevel == self.n_levels - 1:
                break
            x = self.embeddings[ilevel](x, batchidx, condition)

            x, _, _, batchidx, _, _ = self.pools[ilevel](
                x=x.clone(),
                edge_index=torch.empty(2, 0, dtype=torch.long, device=x.device),
                batch=batchidx,
            )
            assert x.shape == (
                conf.loader.batch_size * self.pools[ilevel].ratio,
                self.features[ilevel + 1],
            )

        return x_disc, torch.hstack(x_lat_list)


class BipartPool(nn.Module):
    def __init__(self, in_channels, ratio) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.heads = 4

        self.xcent_base = nn.Parameter(
            torch.normal(0, 1, size=(self.ratio, self.in_channels))
        )

        self.mpl = conv.GATv2Conv(
            in_channels=in_channels,
            out_channels=in_channels,
            heads=self.heads,
            concat=False,
        )

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        batch_size = batch[-1] + 1
        target_graph_size = batch_size * self.ratio
        source_graph_size = len(x)

        source = torch.arange(
            source_graph_size, device=x.device, dtype=torch.long
        ).repeat_interleave(self.ratio)

        target = torch.arange(self.ratio, device=x.device, dtype=torch.long).repeat(
            source_graph_size
        )
        # shift for the batchidx
        target += batch.repeat_interleave(self.ratio) * self.ratio
        # assert len(source) == len(target) == source_graph_size * self.ratio
        # assert ((0 <= source) & (source < source_graph_size)).all()
        # assert ((0 <= target) & (target < target_graph_size)).all()
        # assert max(source) + 1 == source_graph_size
        # assert max(target) + 1 == target_graph_size
        # tcounts = (
        #     target.unique(return_counts=True)[1].reshape(batch_size, self.ratio).T
        # )
        # assert (tcounts[0] == tcounts).all()

        ei_o2c = torch.vstack([source, target])

        xcent = self.mpl(
            x=(x.clone(), self.xcent_base.repeat(batch_size, 1)),
            edge_index=ei_o2c,
        )
        self.batchcent = torch.arange(
            batch_size, device=x.device, dtype=torch.long
        ).repeat_interleave(self.ratio)
        return xcent, None, None, self.batchcent, None, None


class TSumTDisc(nn.Module):
    """Classifies PC via FNN -> Add -> FNN"""

    def __init__(self, n_ftx_out) -> None:
        super().__init__()
        self.disc_emb = nn.ModuleList(
            [
                CentralNodeUpdate(n_ftx_in=n_ftx_out, n_ftx_latent=4, n_global=5)
                for _ in range(2)
            ]
        )
        self.disc = FFN(n_ftx_out, 1, **ffn_param, final_linear=True)

    def forward(self, x, batch):
        for layer in self.disc_emb:
            x = x.clone() + layer(x.clone(), batch)
        x = global_add_pool(x.clone(), batch)
        x = self.disc(x)
        return x


class CentralNodeUpdate(nn.Module):
    """update with global vector"""

    def __init__(self, n_ftx_in, n_ftx_latent, n_global) -> None:
        super().__init__()
        self.emb_nn = FFN(
            n_ftx_in, n_ftx_latent, **(ffn_param | {"norm": "spectral"})
        )
        self.global_nn = FFN(
            n_ftx_latent, n_global, **(ffn_param | {"norm": "spectral"})
        )
        self.out_nn = FFN(
            n_ftx_latent + n_global,
            n_ftx_in,
            **(ffn_param | {"norm": "spectral"}),
            final_linear=True,
        )

    def forward(self, x, batch):
        x = self.emb_nn(x)
        x_aggr = global_add_pool(x, batch)
        x_global = self.global_nn(x_aggr)
        x = self.out_nn(torch.hstack([x, x_global[batch]]))
        return x


class Embedding(nn.Module):
    def __init__(self, n_ftx_in, n_ftx_out, n_ftx_space, n_ftx_latent) -> None:
        super().__init__()
        self.n_ftx_in = n_ftx_in
        self.n_ftx_space = n_ftx_space
        self.n_ftx_latent = n_ftx_latent
        self.n_ftx_out = n_ftx_out

        # self.space_emb = FFN(n_ftx_in, n_ftx_latent, **ffn_param)
        # self.mpls = MPLSeq(
        #     "GINConv",
        #     self.n_ftx_in,
        #     self.n_ftx_latent,
        #     skip_connecton=True,
        #     n_hidden_nodes=self.n_ftx_latent,
        #     layer_param={
        #         k: v for k, v in ffn_param.items() if k != "hidden_layer_size"
        #     }
        #     | {"norm": "batchnorm"},
        #     n_global=0,
        #     n_cond=5,
        #     n_mpl=2,
        # )
        self.inp_emb = FFN(
            self.n_ftx_in,
            self.n_ftx_out,
            **(ffn_param | {"bias": True, "norm": "batchnorm"}),
            final_linear=True,
        )
        self.cnu = CentralNodeUpdate(
            n_ftx_in=self.n_ftx_out,
            n_ftx_latent=self.n_ftx_latent,
            n_global=self.n_ftx_latent,
        )

    def forward(self, x, batch, condition):
        # x = self.space_emb(x)
        # ei = knn_graph(x[..., : self.n_ftx_space], batch=batch, k=5)
        # x = self.mpls(x=x, edge_index=ei, batch=batch, cond=condition)
        x = self.inp_emb(x)
        x = self.cnu(x.clone(), batch) + x.clone()
        return x


def signsqrt(x: torch.Tensor):
    return torch.sign(x) * torch.sqrt(torch.abs(x))


def skipadd(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[:-1] == b.shape[:-1]
    ldim = min(a.shape[-1], b.shape[-1])
    a[..., :ldim] += b[..., :ldim]
    return a
