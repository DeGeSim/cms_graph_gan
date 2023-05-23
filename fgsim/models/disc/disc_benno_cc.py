import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import Parameter

from fgsim.utils.jetnetutils import to_stacked_mask

n_cond = 1


class ModelClass(nn.Module):
    def __init__(self, **kwargs):
        super(ModelClass, self).__init__()
        self.net = Disc(**kwargs)

    def forward(self, batch, cond):
        x = to_stacked_mask(batch)
        x, x_cls = self.net(
            x[..., :-1], mask=(1 - x[..., -1]).bool(), cond=cond[:, [-1]]
        )
        assert not torch.isnan(x).any()
        assert not torch.isnan(x_cls).any()
        return {
            "crit": x.reshape(len(x), -1),
            "latftx": x_cls.reshape(len(x), -1),
        }


class WeightNormalizedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        scale=False,
        bias=False,
        init_factor=1,
        init_scale=1,
    ):
        super(WeightNormalizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        if scale:
            self.scale = Parameter(torch.Tensor(out_features).fill_(init_scale))
        else:
            self.register_parameter("scale", None)

        self.reset_parameters(init_factor)

    def reset_parameters(self, factor):
        stdv = 1.0 * factor / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def weight_norm(self):
        return self.weight.pow(2).sum(1).sqrt().add(1e-8)

    def norm_scale_bias(self, input):
        output = input.div(self.weight_norm().unsqueeze(0))
        if self.scale is not None:
            output = output.mul(self.scale.unsqueeze(0))
        if self.bias is not None:
            output = output.add(self.bias.unsqueeze(0))
        return output

    def forward(self, input):
        return self.norm_scale_bias(F.linear(input, self.weight))

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden, dropout, weightnorm=True):
        super().__init__()
        self.fc0 = (
            WeightNormalizedLinear(embed_dim, hidden)
            if weightnorm
            else nn.Linear(embed_dim, hidden)
        )
        self.fc0_cls = (
            (WeightNormalizedLinear(embed_dim + n_cond, hidden))
            if weightnorm
            else nn.Linear(embed_dim + n_cond, hidden)
        )
        self.fc1 = (
            (WeightNormalizedLinear(hidden + embed_dim, embed_dim))
            if weightnorm
            else nn.Linear(hidden + embed_dim, embed_dim)
        )
        self.fc1_cls = (
            (WeightNormalizedLinear(hidden + 1, embed_dim))
            if weightnorm
            else nn.Linear(hidden + 1, embed_dim)
        )
        self.cond_cls = nn.Linear(1, embed_dim)
        self.attn = weight_norm(
            nn.MultiheadAttention(
                hidden, num_heads, batch_first=True, dropout=dropout
            ),
            "in_proj_weight",
        )
        self.act = nn.LeakyReLU()
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x, x_cls, cond, mask, weight=False):
        res = x.clone()
        res_cls = x_cls.clone()
        x = self.act(self.fc0(x))
        x_cls = self.act(self.ln(self.fc0_cls(torch.cat((x_cls, cond), dim=-1))))
        if weight:
            x_cls, w = self.attn(x_cls, x, x, key_padding_mask=mask)
        else:
            x_cls, w = self.attn(
                x_cls, x, x, key_padding_mask=mask, need_weights=False
            )
        x_cls = self.act(
            self.fc1_cls(
                torch.cat(
                    (x_cls, mask.float().sum(1).unsqueeze(1).unsqueeze(1) / 4000),
                    dim=-1,
                )
            )
        )  # +x.mean(dim=1).
        x_cls = self.act(
            F.glu(torch.cat((x_cls, self.cond_cls(cond[:, :, :1])), dim=-1))
        )
        x = self.act(
            self.fc1(torch.cat((x, x_cls.expand(-1, x.shape[1], -1)), dim=-1)) + res
        )
        x_cls = x_cls + res_cls
        # x=F.glu(torch.cat((x,self.cond(cond[:,:,:1])
        # .expand(-1,x.shape[1],-1)),dim=-1))
        return x, x_cls, w


class Disc(nn.Module):
    def __init__(self, n_dim, l_dim, hidden, num_layers, heads, dropout, **kwargs):
        super().__init__()
        self.embbed = WeightNormalizedLinear(n_dim, l_dim)
        self.encoder = nn.ModuleList(
            [
                Block(
                    embed_dim=l_dim,
                    num_heads=heads,
                    hidden=hidden,
                    dropout=dropout,
                    weightnorm=True,
                )
                for i in range(num_layers)
            ]
        )
        self.out = WeightNormalizedLinear(l_dim, 1)
        self.embbed_cls = WeightNormalizedLinear(l_dim + n_cond, l_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, l_dim), requires_grad=True)
        self.act = nn.LeakyReLU()
        self.fc1 = WeightNormalizedLinear(l_dim, hidden)
        self.fc2 = WeightNormalizedLinear(hidden, l_dim)
        self.ln = nn.LayerNorm(l_dim)

    def forward(self, x, mask, cond, weight=False):  # mean_field=False
        ws = []
        cond = cond.unsqueeze(1)
        x = self.act(self.embbed(x))
        x_cls = torch.cat(
            ((x.sum(1) / 4000).unsqueeze(1).clone(), cond), dim=-1
        )  # self.cls_token.expand(x.size(0), 1, -1)
        x_cls = self.act(self.embbed_cls(x_cls))
        for layer in self.encoder:
            x, x_cls, w = layer(x, x_cls=x_cls, mask=mask, cond=cond, weight=weight)
            res = x_cls.clone()

            x_cls = self.act(x_cls)
            if weight:
                ws.append(w)
        x_cls = self.act(self.ln(self.fc2(self.act(self.fc1(x_cls)))))
        if weight:
            return self.out(x_cls), res, ws
        else:
            return self.out(x_cls), res
