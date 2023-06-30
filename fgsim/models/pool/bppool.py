import torch
from torch import Tensor, nn

from fgsim.config import conf
from fgsim.models.mpl.gatmin import GATv2MinConv


class BipartPool(nn.Module):
    def __init__(self, *, in_channels, ratio, n_heads, mode) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.n_heads = n_heads
        self.mode = mode
        assert self.mode in ["attn", "mpl"]

        self.aggrs = nn.Parameter(
            torch.normal(0, 1, size=(self.ratio, self.in_channels * self.n_heads))
        )

        if self.mode == "attn":
            self.attn = torch.nn.MultiheadAttention(
                embed_dim=self.in_channels * self.n_heads, num_heads=self.n_heads
            )
            mask = ~(
                torch.eye(conf.loader.batch_size)
                .repeat_interleave(self.ratio, 1)
                .bool()
            )
            self.register_buffer("mask", mask, persistent=True)
        else:
            self.mpl = GATv2MinConv(
                in_channels=in_channels * self.n_heads,
                out_channels=in_channels,
                heads=self.n_heads,
                concat=True,
            )
            batchcent = torch.arange(conf.loader.batch_size, dtype=torch.long)
            self.register_buffer("batchcent", batchcent, persistent=True)

    def forward(self, x: Tensor, batch: Tensor):
        batch_size = conf.loader.batch_size
        n_features = x.shape[-1]
        x_aggrs = self.aggrs.repeat(batch_size, 1)

        if self.mode == "attn":
            x_large = x.reshape(-1, n_features)  # self.ln_up()

            attn_output, _ = self.attn(
                x_aggrs, x_large, x_large, attn_mask=self.mask[batch].T
            )
            xcent = attn_output

        else:
            if x.dim() == 3:
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[-1])
            source_graph_size = len(x)

            source = torch.arange(
                source_graph_size, device=x.device, dtype=torch.long
            ).repeat_interleave(self.ratio)

            target = torch.arange(
                self.ratio, device=x.device, dtype=torch.long
            ).repeat(source_graph_size)
            # shift for the batchidx
            target += batch.repeat_interleave(self.ratio) * self.ratio
            # assert len(source) == len(target) == source_graph_size * self.ratio
            # assert ((0 <= source) & (source < source_graph_size)).all()
            # assert ((0 <= target) & (target < target_graph_size)).all()
            # assert max(source) + 1 == source_graph_size
            # assert max(target) + 1 == target_graph_size
            # tcounts = (
            #     target.unique(return_counts=True)[1]
            #     .reshape(batch_size, self.ratio).T
            # )
            # assert (tcounts[0] == tcounts).all()

            ei_o2c = torch.vstack([source, target])

            xcent = self.mpl(
                x=(x, x_aggrs),
                edge_index=ei_o2c,
                # edge_attr=torch.ones_like(source).reshape(-1, 1),
                # size=(x.shape[0], self.aggrs.shape[0] * batch_size),
            )

        return (
            xcent.reshape(batch_size, self.ratio, n_features),
            None,
            None,
            self.batchcent.repeat_interleave(self.ratio),
            None,
            None,
        )
