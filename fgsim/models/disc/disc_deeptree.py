# noqa: F401
import torch
from torch import nn

from fgsim.config import conf
from fgsim.models.common import FFN
from fgsim.models.pool.bppool import BipartPool
from fgsim.models.pool.std_pool import global_mad_pool2 as global_mad_pool


class ModelClass(nn.Module):
    def __init__(
        self,
        *,
        nodes,
        features,
        n_cond,
        cnu_param,
        ffn_param,
        bipart_param,
        emb_param,
        critics_param,
    ):
        super().__init__()
        self.nodes = nodes
        self.features = features
        self.ffn_param = ffn_param
        self.cnu_param = cnu_param
        self.n_cond = n_cond

        self.n_levels = len(self.nodes)
        self.n_heads = bipart_param["n_heads"]
        self.embs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.critics = nn.ModuleList()

        for ilevel in range(self.n_levels):
            self.embs.append(
                Embedding(
                    n_ftx_in=(
                        self.features[ilevel] * self.n_heads
                        if ilevel != 0
                        else self.features[ilevel]
                    ),
                    n_ftx_out=self.features[ilevel + 1] * self.n_heads,
                    ffn_param=ffn_param,
                    cnu_param=cnu_param,
                    **emb_param,
                )
            )
            self.pools.append(
                BipartPool(
                    in_channels=self.features[ilevel + 1],
                    ratio=self.nodes[ilevel],
                    **bipart_param,
                    batch_size=conf.loader.batch_size,
                )
            )
        for ilevel in range(self.n_levels + 1):
            self.critics.append(
                TSumTDisc(
                    n_ftx=(
                        self.features[ilevel] * self.n_heads
                        if ilevel != 0
                        else self.features[ilevel]
                    ),
                    n_cond=n_cond,
                    ffn_param=ffn_param,
                    cnu_param=cnu_param,
                    **critics_param,
                )
            )

    def forward(self, batch, condition):
        x: torch.Tensor
        x, batchidx = batch.x, batch.batch

        x_lat_dict = {"input": torch.hstack(global_mad_pool(x, batchidx)[1:])}
        score_List = []

        for ilevel in range(self.n_levels + 1):
            assert not x.isnan().any()

            score_List.append(self.critics[ilevel](x, batchidx, condition).clone())

            if ilevel == self.n_levels:
                break
            x = self.embs[ilevel](x, batchidx)
            # aggregate latent space features
            x_lat_dict[f"lvl{ilevel}_emb"] = x

            x, _, _, batchidx, _, _ = self.pools[ilevel](
                x=x.clone(), batch=batchidx
            )
            x_lat_dict[f"lvl{ilevel}_pool"] = x

            assert x.shape == (
                conf.loader.batch_size,
                self.pools[ilevel].ratio,
                self.features[ilevel + 1] * self.n_heads,
            )

        return {
            "crit": torch.vstack(score_List),
            "latftx": x_lat_dict,
        }


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
        cnu_param,
        ffn_param,
    ) -> None:
        super().__init__()
        self.emb = nn.ModuleList(
            [
                CentralNodeUpdate(
                    n_ftx_in=n_ftx,
                    n_ftx_latent=n_ftx,
                    n_global=n_ftx_global,
                    **cnu_param,
                    ffn_param=ffn_param,
                )
                for _ in range(n_updates)
            ]
        )
        self.red = FFN(1 + 3 * n_ftx + n_cond, 1, **ffn_param, final_linear=True)

    def forward(self, x, batch, condition):
        for layer in self.emb:
            x = x.clone() + layer(x.clone(), batch)
        x = global_mad_pool(x.clone(), batch)
        x = self.red(torch.hstack([*x, condition]))
        return x


class CentralNodeUpdate(nn.Module):
    """update with global vector"""

    def __init__(self, n_ftx_in, n_ftx_latent, norm, n_global, ffn_param) -> None:
        super().__init__()
        ffn_param = ffn_param | {"norm": norm}
        self.emb = FFN(n_ftx_in, n_ftx_latent, **ffn_param)
        self.glob = FFN(1 + n_ftx_latent * 3, n_global, **ffn_param)
        self.out = FFN(
            n_ftx_latent + n_global, n_ftx_in, final_linear=True, **ffn_param
        )

    def forward(self, x, batch=None):
        x = self.emb(x)

        x_glob = torch.hstack(global_mad_pool(x, batch))

        x_global = self.glob(x_glob)
        if x.dim() == 2:
            ten_l = [x, x_global[batch]]
        else:
            ten_l = [x, x_global[batch].reshape(x.shape[0], x.shape[1], -1)]

        x = self.out(torch.concat(ten_l, dim=-1))
        return x


class Embedding(nn.Module):
    def __init__(
        self, *, n_ftx_in, n_ftx_out, n_ftx_latent, norm, cnu_param, ffn_param
    ) -> None:
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
        self.inemb = FFN(
            self.n_ftx_in,
            self.n_ftx_out,
            **(ffn_param | {"bias": True, "norm": norm}),
            final_linear=True,
        )
        self.cnu = CentralNodeUpdate(
            n_ftx_in=self.n_ftx_out,
            n_ftx_latent=self.n_ftx_latent,
            n_global=self.n_ftx_latent,
            **cnu_param,
            ffn_param=ffn_param,
        )

    def forward(self, x, batch):
        # x = self.space_emb(x)
        # ei = knn_graph(x[..., : self.n_ftx_space], batch=batch, k=5)
        # x = self.mpls(x=x, edge_index=ei, batch=batch, cond=condition)
        x = self.inemb(x)
        x = self.cnu(x.clone(), batch) + x.clone()
        return x


def signsqrt(x: torch.Tensor):
    return torch.sign(x) * torch.sqrt(torch.abs(x))


def skipadd(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[:-1] == b.shape[:-1]
    ldim = min(a.shape[-1], b.shape[-1])
    a[..., :ldim] += b[..., :ldim]
    return a
