from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://proceedings.mlr.press/v80/achlioptas18a.html
# https://github.com/optas/latent_3d_points


class ModelClass(nn.Module):
    def __init__(self, features: List[int]):
        features = features[::-1]

        self.layer_num = len(features) - 1
        super().__init__()

        self.fc_layers = nn.ModuleList(
            [
                nn.Conv1d(features[inx], features[inx + 1], kernel_size=1, stride=1)
                for inx in range(self.layer_num)
            ]
        )

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(
            nn.Linear(features[-1], features[-2]),
            nn.Linear(features[-2], features[0]),
            nn.Linear(features[0], 1),
        )

    def forward(self, batch):
        n_features = batch.x.shape[1]
        batch_size = max(batch.batch) + 1

        f = batch.x.reshape(batch_size, -1, n_features)
        # # check if the reshape worked as expected:
        # f_slow = torch.stack(
        #     [
        #         batch.x[batch.ptr[ibatch] : batch.ptr[ibatch + 1]]
        #         for ibatch in range(batch_size)
        #     ]
        # )
        # assert torch.all(f == f_slow)

        # expected shape here: (batch_size,points,features)
        # transpose to do convolutions on the features
        feat = f.transpose(1, 2)
        max_hits = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=max_hits).squeeze(-1)
        out = self.final_layer(out).squeeze()  # (B, 1)
        assert not torch.any(torch.isnan(out))
        return out
