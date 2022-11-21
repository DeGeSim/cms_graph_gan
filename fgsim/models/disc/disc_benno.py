import torch
from torch import nn
from torch.nn.functional import leaky_relu

from fgsim.utils.jetnetutils import to_stacked_mask


class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        self.net = Disc()

    def forward(self, batch, cond):
        x = to_stacked_mask(batch)
        x = self.net(x[..., :3], mask=(1 - x[..., 3]).bool())
        return x


class Disc(nn.Module):
    def __init__(
        self,
        n_dim=3,
        l_dim=25,
        hidden=512,
        num_layers=4,
        num_heads=5,
        n_part=30,
        dropout=0.05,
    ):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        self.l_dim = l_dim
        self.embbed = nn.Linear(n_dim, l_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.l_dim,
                nhead=num_heads,
                dim_feedforward=hidden,
                dropout=dropout,
                norm_first=False,
                activation=lambda x: leaky_relu(x, 0.2),
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.encoder_class = nn.TransformerEncoderLayer(
            d_model=self.l_dim,
            nhead=num_heads,
            dim_feedforward=hidden,
            dropout=dropout,
            norm_first=False,
            activation=lambda x: leaky_relu(x, 0.2),
            batch_first=True,
        )
        self.hidden = nn.Linear(l_dim, 2 * hidden)
        self.hidden2 = nn.Linear(2 * hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, m=None, p=None, mask=None, noise=0):
        x = self.embbed(x)
        mask = (
            torch.concat(
                (torch.zeros_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1
            )
            .to(x.device)
            .bool()
        )
        x = torch.concat(
            (torch.zeros_like(x[:, 0, :]).reshape(len(x), 1, -1), x), axis=1
        )
        x = self.encoder(x, src_key_padding_mask=mask.bool())
        x = x[:, 0, :]
        x = leaky_relu(self.hidden(x), 0.2)
        x = leaky_relu(self.hidden2(x), 0.2)
        x = self.out(x)
        return x
