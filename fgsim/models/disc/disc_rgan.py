import torch
from torch import nn


class ModelClass(torch.nn.Module):
    def __init__(self, rgand_sfc, leaky_relu_alpha, rgand_fc):
        super(ModelClass, self).__init__()
        self.rgand_sfc = rgand_sfc
        self.rgand_fc = rgand_fc

        self.rgand_sfc.insert(0, self.node_feat_size)

        layers = []
        for i in range(len(self.rgand_sfc) - 1):
            layers.append(nn.Conv1d(self.rgand_sfc[i], self.rgand_sfc[i + 1], 1))
            layers.append(nn.LeakyReLU(negative_slope=leaky_relu_alpha))

        self.sfc = nn.Sequential(*layers)

        self.rgand_fc.insert(0, self.rgand_sfc[-1])

        layers = []
        for i in range(len(self.rgand_fc) - 1):
            layers.append(nn.Linear(self.rgand_fc[i], self.rgand_fc[i + 1]))
            layers.append(nn.LeakyReLU(negative_slope=leaky_relu_alpha))

        layers.append(nn.Linear(self.rgand_fc[-1], 1))
        layers.append(nn.Sigmoid())

        self.fc = nn.Sequential(*layers)

    def forward(self, x, labels=None, epoch=None):
        x = x.reshape(-1, self.node_feat_size, 1)
        x = self.sfc(x)
        x = torch.max(x.reshape(-1, self.num_hits, self.rgand_sfc[-1]), 1)[0]
        return self.fc(x)
