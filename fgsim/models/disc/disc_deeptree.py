# noqa: F401
import torch
from torch import Tensor, nn
from torch_geometric.nn import conv, global_add_pool, global_max_pool

from fgsim.config import conf
from fgsim.models.common import FFN
from fgsim.utils.std_pool import global_width_pool


class ModelClass(nn.Module):
    def __init__(
        self,
        *,
        nodes,
        features,
        n_cond,
        ffn_param,
        bipart_param,
        emb_param,
        critics_param,
    ):
        super().__init__()
        self.nodes = nodes
        self.features = features
        self.ffn_param = ffn_param
        self.n_cond = n_cond

        self.n_levels = len(self.nodes)

        self.embeddings = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.pcdiscs = nn.ModuleList()

        for ilevel in range(self.n_levels):
            self.embeddings.append(
                Embedding(
                    n_ftx_in=self.features[ilevel],
                    n_ftx_out=self.features[ilevel + 1],
                    ffn_param=ffn_param,
                    **emb_param,
                )
            )
            self.pools.append(
                BipartPool(
                    in_channels=self.features[ilevel + 1],
                    ratio=self.nodes[ilevel],
                    **bipart_param,
                )
            )
        for ilevel in range(self.n_levels + 1):
            self.pcdiscs.append(
                TSumTDisc(
                    n_ftx=self.features[ilevel],
                    n_cond=n_cond,
                    ffn_param=ffn_param,
                    **critics_param,
                )
            )

    def forward(self, batch, condition):
        x: torch.Tensor
        x, batchidx = batch.x, batch.batch
        x_disc = torch.zeros((batch.num_graphs, 1), dtype=x.dtype, device=x.device)

        x_lat_list = []
        cond_reg_list = []

        for ilevel in range(self.n_levels + 1):
            assert not x.isnan().any()

            # aggregate latent space features
            for f in [global_add_pool, global_max_pool, global_width_pool]:
                lat_aggr = f(x.clone(), batchidx)
                # assert not lat_aggr.isnan().any()
                x_lat_list.append(lat_aggr)

            d, cond_reg = self.pcdiscs[ilevel](x.clone(), batchidx)
            # assert not d.isnan().any()
            # assert not cond_reg.isnan().any()
            x_disc += d
            cond_reg_list.append(cond_reg)

            if ilevel == self.n_levels:
                break
            x = self.embeddings[ilevel](x.clone(), batchidx)
            # assert not x.isnan().any()

            x, _, _, batchidx, _, _ = self.pools[ilevel](
                x=x.clone(),
                edge_index=torch.empty(2, 0, dtype=torch.long, device=x.device),
                batch=batchidx,
            )
            # assert not x.isnan().any()
            assert x.shape == (
                conf.loader.batch_size * self.pools[ilevel].ratio,
                self.features[ilevel + 1],
            )

        return {
            "crit": x_disc,
            "latftx": torch.hstack(x_lat_list),
            "condreg": torch.stack(cond_reg_list).mean(dim=0),
        }


class BipartPool(nn.Module):
    def __init__(self, *, in_channels, ratio, n_heads) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.n_heads = n_heads

        self.xcent_base = nn.Parameter(
            torch.normal(0, 1, size=(self.ratio, self.in_channels))
        )

        self.mpl = conv.GATv2Conv(
            in_channels=in_channels,
            out_channels=in_channels,
            heads=self.n_heads,
            concat=False,
        )

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        batch_size = int(batch[-1] + 1)
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
            # edge_attr=torch.ones_like(source).reshape(-1, 1),
            # size=(x.shape[0], self.xcent_base.shape[0] * batch_size),
        )
        self.batchcent = torch.arange(
            batch_size, device=x.device, dtype=torch.long
        ).repeat_interleave(self.ratio)
        return xcent, None, None, self.batchcent, None, None


class TSumTDisc(nn.Module):
    """Classifies PC via FNN -> Add -> FNN"""

    def __init__(
        self,
        *,
        n_ftx,
        n_ftx_latent,
        n_cond,
        n_ftx_global,
        n_updates,
        ffn_param,
    ) -> None:
        super().__init__()
        self.disc_emb = nn.ModuleList(
            [
                CentralNodeUpdate(
                    n_ftx_in=n_ftx,
                    n_ftx_latent=n_ftx_latent,
                    n_global=n_ftx_global,
                    ffn_param=ffn_param,
                )
                for _ in range(n_updates)
            ]
        )
        self.aggrs = [global_max_pool, global_add_pool]
        self.disc = FFN(
            n_ftx * len(self.aggrs), 1 + n_cond, **ffn_param, final_linear=True
        )

    def forward(self, x, batch):
        for layer in self.disc_emb:
            x = x.clone() + layer(x.clone(), batch)
        x = self.disc(torch.hstack([f(x.clone(), batch) for f in self.aggrs]))
        return (
            x[:, :1],
            x[:, 1:],
        )


class CentralNodeUpdate(nn.Module):
    """update with global vector"""

    def __init__(self, n_ftx_in, n_ftx_latent, n_global, ffn_param) -> None:
        super().__init__()
        self.emb_nn = FFN(
            n_ftx_in, n_ftx_latent, **(ffn_param | {"norm": "spectral"})
        )
        self.aggrs = [global_max_pool, global_add_pool]
        self.global_nn = FFN(
            len(self.aggrs) * n_ftx_latent,
            n_global,
            **(ffn_param | {"norm": "spectral"}),
        )

        self.out_nn = FFN(
            n_ftx_latent + n_global,
            n_ftx_in,
            **(ffn_param | {"norm": "spectral"}),
            final_linear=True,
        )

    def forward(self, x, batch):
        x = self.emb_nn(x)
        # assert not x.isnan().any()
        x_aggr = torch.hstack([f(x.clone(), batch) for f in self.aggrs])
        # assert not x_aggr.isnan().any()
        x_global = self.global_nn(x_aggr)
        x = self.out_nn(torch.hstack([x, x_global[batch]]))
        # assert not x.isnan().any()
        return x


class Embedding(nn.Module):
    def __init__(self, *, n_ftx_in, n_ftx_out, n_ftx_latent, ffn_param) -> None:
        super().__init__()
        self.n_ftx_in = n_ftx_in
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
            ffn_param=ffn_param,
        )

    def forward(self, x, batch):
        # x = self.space_emb(x)
        # ei = knn_graph(x[..., : self.n_ftx_space], batch=batch, k=5)
        # x = self.mpls(x=x, edge_index=ei, batch=batch, cond=condition)
        # assert not x.isnan().any()
        x_post = self.inp_emb(x)
        # assert not x_post.isnan().any()
        x = self.cnu(x_post.clone(), batch) + x_post.clone()
        # assert not x.isnan().any()
        return x


def signsqrt(x: torch.Tensor):
    return torch.sign(x) * torch.sqrt(torch.abs(x))


def skipadd(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[:-1] == b.shape[:-1]
    ldim = min(a.shape[-1], b.shape[-1])
    a[..., :ldim] += b[..., :ldim]
    return a
