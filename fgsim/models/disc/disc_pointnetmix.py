from typing import List

import torch
from torch import nn

from fgsim.config import conf


class ModelClass(torch.nn.Module):
    def __init__(
        self,
        pointnetd_pointfc: List,
        pointnetd_fc: List,
        node_feat_size: int,
        leaky_relu_alpha: float,
        num_hits: int,
        mask: bool,
    ):
        super(ModelClass, self).__init__()
        self.pointnetd_pointfc: List = pointnetd_pointfc
        self.pointnetd_fc: List = pointnetd_fc
        self.node_feat_size: int = node_feat_size
        self.leaky_relu_alpha: float = leaky_relu_alpha
        self.num_hits: int = num_hits
        self.mask: bool = mask

        self.pointnetd_pointfc.insert(0, self.node_feat_size)
        self.pointnetd_fc.insert(0, self.pointnetd_pointfc[-1] * 2)
        self.pointnetd_fc.append(1)

        layers: List[nn.Module] = []

        for i in range(len(self.pointnetd_pointfc) - 1):
            layers.append(
                nn.Linear(
                    self.pointnetd_pointfc[i],
                    self.pointnetd_pointfc[i + 1],
                )
            )
            layers.append(nn.LeakyReLU(negative_slope=leaky_relu_alpha))

        self.pointfc = nn.Sequential(*layers)

        layers = []

        for i in range(len(self.pointnetd_fc) - 1):
            layers.append(nn.Linear(self.pointnetd_fc[i], self.pointnetd_fc[i + 1]))
            if i < len(self.pointnetd_fc) - 2:
                layers.append(nn.LeakyReLU(negative_slope=leaky_relu_alpha))

        layers.append(nn.Sigmoid())
        self.fc = nn.Sequential(*layers)

    def forward(self, batch):
        x = batch.x
        n_points = conf.loader.n_points
        batch_size = x.shape[0] // n_points
        # n_features = conf.loader.n_features
        # add the ones mask
        # x = torch.hstack((x, torch.ones(batch_size * n_points, 1, device=x.device)))
        # x = x.reshape(batch_size, n_points, n_features + 1)

        # batch_size = x.size(0)
        if self.mask:
            x[:, :, 2] += 0.5
            mask = x[:, :, 3:4] >= 0
            x = (x * mask)[:, :, :3]
            x[:, :, 2] -= 0.5
        x = self.pointfc(
            x.view(batch_size * self.num_hits, self.node_feat_size)
        ).view(batch_size, self.num_hits, self.pointnetd_pointfc[-1])
        x = torch.cat((torch.max(x, dim=1)[0], torch.mean(x, dim=1)), dim=1)
        return self.fc(x)
