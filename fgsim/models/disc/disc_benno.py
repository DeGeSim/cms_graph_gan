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
        x = self.net(x[..., :3], mask=(1 - x[..., 3]))
        return x


class Disc(nn.Module):
    def __init__(
        self,
        n_dim=3,
        l_dim=10,
        hidden=300,
        num_layers=3,
        num_heads=1,
        n_part=2,
        fc=False,
        dropout=0.5,
        mass=False,
        clf=False,
    ):
        super().__init__()
        self.hidden_nodes = hidden
        self.n_dim = n_dim
        #         l_dim=n_dim
        self.l_dim = l_dim
        self.n_part = n_part
        self.fc = fc
        self.clf = clf

        if fc:
            self.l_dim *= n_part
            self.embbed_flat = nn.Linear(n_dim * n_part, l_dim)
            self.flat_hidden = nn.Linear(l_dim, hidden)
            self.flat_hidden2 = nn.Linear(hidden, hidden)
            self.flat_hidden3 = nn.Linear(hidden, hidden)
            self.flat_out = nn.Linear(hidden, 1)
        else:
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
            self.hidden = nn.Linear(l_dim + int(mass), 2 * hidden)
            self.hidden2 = nn.Linear(2 * hidden, hidden)
            self.out = nn.Linear(hidden, 1)

    def forward(self, x, m=None, mask=None):
        if self.fc:
            x = x.reshape(len(x), self.n_dim * self.n_part)
            x = self.embbed_flat(x)
            x = leaky_relu(self.flat_hidden(x), 0.2)
            x = leaky_relu(self.flat_hidden2(x), 0.2)
            x = self.flat_out(x)
        else:
            x = self.embbed(x)
            if self.clf:
                x = torch.concat(
                    (torch.ones_like(x[:, 0, :]).reshape(len(x), 1, -1), x), axis=1
                )
                mask = torch.concat(
                    (torch.ones_like((mask[:, 0]).reshape(len(x), 1)), mask), dim=1
                ).to(x.device)

                x = self.encoder(x, src_key_padding_mask=mask)
                x = x[:, 0, :]
            else:
                x = self.encoder(x, src_key_padding_mask=mask)
                x = torch.sum(x, axis=1)
            if m is not None:
                x = torch.concat((m.reshape(len(x), 1), x), axis=1)
            x = leaky_relu(self.hidden(x), 0.2)
            x = leaky_relu(self.hidden2(x), 0.2)
            x = self.out(x)
            x = x
        return x
