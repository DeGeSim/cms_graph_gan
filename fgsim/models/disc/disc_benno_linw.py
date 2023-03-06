import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter

from fgsim.utils.jetnetutils import to_stacked_mask


class ModelClass(nn.Module):
    def __init__(self, **kwargs):
        super(ModelClass, self).__init__()
        self.net = Disc(**kwargs)

    def forward(self, batch, cond):
        x = to_stacked_mask(batch)
        x = self.net(x[..., :3], mask=(1 - x[..., 3]).bool())
        return x


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


class BlockCls(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden):
        super().__init__()
        self.fc0 = WeightNormalizedLinear(embed_dim, hidden)
        self.fc1_cls = WeightNormalizedLinear(embed_dim, hidden)
        self.fc2_cls = WeightNormalizedLinear(hidden, embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0
        )
        self.act = nn.LeakyReLU()
        # self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x, x_cls, src_key_padding_mask=None):
        res = x_cls.clone()
        x = self.act(self.fc0(x))
        x_cls = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[0]
        x = x + x_cls
        # x= self.bn(x.reshape(-1,x.shape[-1])).reshape(x.shape)
        x_cls = self.act(self.fc1_cls(x_cls))  # +x.mean(dim=1).unsqueeze(1)
        x_cls = self.act(self.fc2_cls(x_cls + res))
        return x_cls, x


class Disc(nn.Module):
    def __init__(
        self, n_dim, l_dim, hidden, num_layers, heads, mean_field, **kwargs
    ):
        super().__init__()
        l_dim = hidden
        self.embbed = WeightNormalizedLinear(n_dim, l_dim)
        self.encoder = nn.ModuleList(
            [
                BlockCls(embed_dim=l_dim, num_heads=heads, hidden=hidden)
                for i in range(num_layers)
            ]
        )
        self.out = WeightNormalizedLinear(l_dim, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim), requires_grad=True)
        self.act = nn.LeakyReLU()
        self.fc1 = WeightNormalizedLinear(l_dim, hidden)
        self.mean_field = mean_field
        # self.fc2 = WeightNormalizedLinear(hidden, l_dim)
        self.apply(self._init_weights)

    def forward(self, x, mask=None):
        x = self.act(self.embbed(x))
        x_cls = self.cls_token.expand(x.size(0), 1, -1)
        for layer in self.encoder:
            x_cls, x = layer(x, x_cls=x_cls, src_key_padding_mask=mask)
        res = x_cls.clone()
        x_cls = self.act(self.fc1(x_cls))
        if self.mean_field:
            return self.out(x_cls).squeeze(-1), res
        else:
            return self.out(x_cls).squeeze(-1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight,
            )
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.MultiheadAttention):
            nn.init.kaiming_normal_(m.in_proj_weight)
        torch.nn.init.kaiming_normal_(
            self.embbed.weight,
        )
        torch.nn.init.kaiming_normal_(
            self.out.weight,
        )
