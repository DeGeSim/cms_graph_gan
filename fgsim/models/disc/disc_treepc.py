from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from fgsim.utils.jetnetutils import to_stacked_mask

# https://proceedings.mlr.press/v80/achlioptas18a.html
# https://github.com/optas/latent_3d_points


class ModelClass(nn.Module):
    def __init__(self, features: List[int]):
        self.features = features[::-1]

        self.layer_num = len(self.features) - 1
        super().__init__()

        self.fc_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    self.features[inx],
                    self.features[inx + 1],
                    kernel_size=1,
                    stride=1,
                )
                for inx in range(self.layer_num)
            ]
        )

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(
            nn.Linear(self.features[-1], self.features[-2]),
            nn.Linear(self.features[-2], self.features[0]),
            nn.Linear(self.features[0], 1),
        )

    def forward(self, batch, cond):
        n_features = batch.x.shape[1]
        batch_size = batch.batch[-1] + 1
        x = to_stacked_mask(batch)[..., : self.features[0]]
        f = x.reshape(batch_size, -1, n_features)
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
