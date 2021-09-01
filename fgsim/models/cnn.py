"""
The CNN model from https://github.com/MattUnderscoreZhang/Triforce_CaloML/\
blob/master/Training/TriForce/Architectures/GoogLeNet.py
"""


import torch
import torch.nn.functional as F
from torch import nn

from fgsim.config import conf, device

nfeatures = conf.model.dyn_features + conf.model.static_features
n_hl_features = len(conf.loader.keylist) - 2 - 1

epsilon = 1e-07
CLASSIFICATION, REGRESSION = 0, 1


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):

        super().__init__()

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv3d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm3d(n1x1, eps=epsilon),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv3d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm3d(n3x3red, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(n3x3, eps=epsilon),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv3d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm3d(n5x5red, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5, eps=epsilon),
            nn.ReLU(True),
            nn.Conv3d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(n5x5, eps=epsilon),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool3d(3, stride=1, padding=1),
            nn.Conv3d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm3d(pool_planes, eps=epsilon),
            nn.ReLU(True),
        )

    def forward(self, x):

        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class ModelClass(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_layers = nn.Sequential(
            nn.Conv3d(1, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192, eps=epsilon),
            nn.ReLU(True),
        )

        self.norm = nn.InstanceNorm3d(1)
        # self.norm = nn.BatchNorm3d(1)

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool3d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool3d(7, stride=1)
        self.dense = nn.Linear(
            1024 + 4, 1024
        )  # window size of 25, plus reco angles and energy sums
        self.linear = nn.Linear(1024 + 5, 1)  # output layer

    def forward(self, batch):
        ECAL = batch["ECAL"]
        ECAL_sum = batch["ECAL_sum"]
        HCAL_sum = batch["HCAL_sum"]
        recoEta = batch["recoEta"]
        recoPhi = batch["recoPhi"]

        # net
        x = self.norm(ECAL)
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)
        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)
        x = self.a5(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # cat angles / energy sums in before dense layer
        x = torch.cat([x, recoPhi, recoEta, ECAL_sum, HCAL_sum], 1)
        x = F.relu(self.dense(x))
        # cat angles / energy sums back in before final layer
        x = torch.cat(
            [
                x,
                recoPhi,
                recoEta,
                ECAL_sum,
                HCAL_sum,
                torch.ones([batch["ECAL"].shape[0], 1], device=device),
            ],
            1,
        )
        x = self.linear(x)
        # preparing output
        return x
