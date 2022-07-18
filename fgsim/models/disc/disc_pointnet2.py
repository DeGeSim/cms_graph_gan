# https://github.com/charlesq34/pointnet
# / https://github.com/fxia22/pointnet.pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable


class ModelClass(nn.Module):
    def __init__(self, batch_size: int, n_points: int, n_features: int):
        super().__init__()
        self.batch_size = batch_size
        self.n_points = n_points
        self.n_features = n_features
        self.net = PointNetCls(
            n_features=n_features, n_classes=2, feature_transform=True
        )

    def forward(self, batch):
        x = batch.x[torch.argsort(batch.batch)].reshape(
            self.batch_size, self.n_points, self.n_features
        )
        x = x.transpose(1, 2)
        out = self.net(x)
        return out


class PointNetCls(nn.Module):
    def __init__(self, n_features, n_classes, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(
            input_features=n_features,
            global_feat=True,
            feature_transform=feature_transform,
        )
        if n_classes == 2:  # if binary then dim=1
            n_classes = 1
        self.n_classes = n_classes
        self.ffn = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.ffn(x)
        if self.n_classes == 1:
            return x.squeeze()
        else:
            return F.log_softmax(x, dim=1).squeeze()  # , trans, trans_feat


class PointNetfeat(nn.Module):
    def __init__(self, input_features, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(input_features)

        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(64)
        self.seq1 = nn.Sequential(
            torch.nn.Conv1d(input_features, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        self.seq2 = nn.Sequential(
            torch.nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            torch.nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
        )

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = self.seq1(x)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.seq2(x)
        x = torch.sum(x, 2, keepdim=True)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    # batchsize = trans.size()[0]
    Idt = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        Idt = Idt.cuda()
    loss = torch.mean(
        torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - Idt, dim=(1, 2))
    )
    return loss


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k

        self.seq1 = nn.Sequential(
            torch.nn.Conv1d(k, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            torch.nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            torch.nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.seq2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.seq1(x)
        x = torch.sum(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        x = self.seq2(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x
