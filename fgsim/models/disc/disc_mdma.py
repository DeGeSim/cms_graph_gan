import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm

from fgsim.models.common.benno import WeightNormalizedLinear
from fgsim.utils.jetnetutils import to_stacked_mask

n_cond = 1


class ModelClass(nn.Module):
    def __init__(self, **kwargs):
        super(ModelClass, self).__init__()
        self.net = Disc(**kwargs)

    def forward(self, batch, cond):
        x = to_stacked_mask(batch)
        x, x_cls = self.net(x[..., :-1], mask=(1 - x[..., -1]).bool(), cond=cond)
        assert not torch.isnan(x).any()
        assert not torch.isnan(x_cls).any()
        return {
            "crit": x.reshape(len(x), -1),
            "latftx": x_cls.reshape(len(x), -1),
        }


class BlockCls(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden, dropout):
        super().__init__()

        self.fc0 = WeightNormalizedLinear(embed_dim, hidden)
        self.fc1 = WeightNormalizedLinear(hidden + embed_dim, embed_dim)
        self.fc1_cls = WeightNormalizedLinear(hidden + 1, embed_dim)
        self.attn = weight_norm(
            nn.MultiheadAttention(
                hidden, num_heads, batch_first=True, dropout=dropout
            ),
            "in_proj_weight",
        )

        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)
        self.hidden = hidden

    def forward(self, x, x_cls, mask, weight=False):
        res = x_cls.clone()
        x = self.act(self.fc0(x))
        x_cls = self.act(self.ln(self.fc0(x_cls)))
        if weight:
            x_cls, w = self.attn(x_cls, x, x, key_padding_mask=mask)
        else:
            x_cls = self.attn(x_cls, x, x, key_padding_mask=mask)[0]
            w = None
        x_cls = self.act(
            self.fc1_cls(
                torch.cat(
                    (x_cls, mask.sum(1).unsqueeze(1).unsqueeze(1) / 70), dim=-1
                )
            )
        )
        x = self.act(
            self.fc1(torch.cat((x, x_cls.expand(-1, x.shape[1], -1)), dim=-1))
        )
        x_cls = x_cls + res
        return x_cls, x, w


class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads, dropout, **kwargs):
        super().__init__()
        self.embbed = WeightNormalizedLinear(n_dim, l_dim)
        self.encoder = nn.ModuleList(
            [
                BlockCls(
                    embed_dim=l_dim, num_heads=heads, hidden=hidden, dropout=dropout
                )
                for i in range(num_layers)
            ]
        )
        self.out = WeightNormalizedLinear(l_dim, 1)
        self.embbed_cls = WeightNormalizedLinear(l_dim + 1, l_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, l_dim), requires_grad=True)
        self.act = nn.LeakyReLU()
        self.fc1 = WeightNormalizedLinear(l_dim, hidden)
        self.fc2 = WeightNormalizedLinear(hidden, l_dim)
        self.ln = nn.LayerNorm(l_dim)

    def forward(self, x, mask, cond, weight=False):  # mean_field=False
        ws = []
        x = self.act(self.embbed(x))
        x_cls = torch.cat(
            (
                x.mean(1).unsqueeze(1).clone(),
                mask.sum(1).unsqueeze(1).unsqueeze(1) / 70,
            ),
            dim=-1,
        )  # self.cls_token.expand(x.size(0), 1, -1)
        x_cls = self.act(self.embbed_cls(x_cls))
        for layer in self.encoder:
            x_cls, x, w = layer(x, x_cls=x_cls, mask=mask, weight=weight)
            res = x_cls.clone()
            x_cls = self.act(x_cls)
            if weight:
                ws.append(w)
        x_cls = self.act(self.ln(self.fc2(self.act(self.fc1(x_cls)))))
        if weight:
            return self.out(x_cls), res, ws
        else:
            return self.out(x_cls), res
