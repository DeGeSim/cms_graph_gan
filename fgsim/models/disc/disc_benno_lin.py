import torch
from torch import nn
from torch.nn.functional import leaky_relu as leaky
from torch.nn.utils.parametrizations import spectral_norm

from fgsim.utils.jetnetutils import to_stacked_mask


class ModelClass(nn.Module):
    def __init__(self, **kwargs):
        super(ModelClass, self).__init__()
        self.net = Disc(**kwargs)

    def forward(self, batch, cond):
        x = to_stacked_mask(batch)
        x = self.net(x[..., :3], mask=(1 - x[..., 3]).bool())
        return x


class BlockCls(nn.Module):
    def __init__(
        self, embed_dim, num_heads, hidden, dropout, activation, slope, norm
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = hidden
        self.reduce_embbed = nn.Linear(2 * embed_dim, embed_dim)
        self.slope = slope
        self.norm = norm
        if self.norm:
            self.pre_attn_norm = nn.LayerNorm(
                embed_dim,
            )
            self.pre_fc_norm = nn.LayerNorm(
                embed_dim,
            )
        self.attn = spectral_norm(
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
            "in_proj_weight",
        )
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.GroupNorm(
            1,
            embed_dim,
        )
        self.fc1 = spectral_norm(nn.Linear(embed_dim, self.ffn_dim))
        self.act = nn.GELU() if activation == "gelu" else nn.LeakyReLU()
        self.act_dropout = nn.Dropout(dropout)
        self.fc2 = spectral_norm(nn.Linear(self.ffn_dim, embed_dim))

        self.hidden = hidden

    def forward(self, x, x_cls=None, src_key_padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.
        Returns:
            encoded output of shape `(batch,seq_len, embed_dim)`
        """

        residual = x_cls.clone()

        # if self.norm:
        #     u = self.pre_attn_norm(x)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        x_cls = self.attn(x_cls, x, x, key_padding_mask=src_key_padding_mask)[
            0
        ]  # ( batch,1, embed_dim)

        # x=self.reduce_embbed(torch.cat((x_cls,x),axis=-1))
        # x = (x+ residual)
        # residual = x

        # if self.norm:
        #     x = self.pre_fc_norm(x)
        x_cls = leaky(self.fc1(x_cls), self.slope)
        x_cls = self.act_dropout(x_cls)
        x_cls = self.fc2(x_cls) + residual

        return x_cls


class Disc(nn.Module):
    def __init__(
        self,
        n_dim,
        l_dim,
        hidden,
        num_layers,
        heads,
        dropout,
        activation,
        slope,
        norm,
        **kwargs,
    ):
        super().__init__()
        self.embbed = nn.Linear(n_dim, l_dim)
        self.slope = slope
        self.encoder = nn.ModuleList(
            [
                BlockCls(
                    embed_dim=l_dim,
                    num_heads=heads,
                    hidden=int(hidden),
                    dropout=dropout,
                    activation=activation,
                    slope=0.1,
                    norm=norm,
                )
                for i in range(num_layers)
            ]
        )

        self.hidden = spectral_norm(nn.Linear(l_dim, hidden))
        self.hidden2 = spectral_norm(nn.Linear(int(hidden), l_dim))
        self.out = nn.Linear(l_dim, 1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, l_dim), requires_grad=True)

    def forward(self, x, mask=None):
        x = self.embbed(x)
        x_cls = self.cls_token.expand(x.size(0), 1, -1).clone()
        for layer in self.encoder:
            x_cls = layer(
                x, x_cls, src_key_padding_mask=mask.bool()
            )  # attention_mask.bool()
        x = x_cls.reshape(len(x), x.shape[-1])

        x = leaky(self.hidden(x), self.slope)
        x = leaky(self.hidden2(x), self.slope)
        # x = leaky(self.hidden2(x), self.slope)
        # x = leaky(self.batchnorm(self.hidden(x)), self.slope)
        # x = leaky(self.batchnorm2(self.hidden2(x)), self.slope)

        return self.out(x)
