import torch.nn as nn

from fgsim.config import conf
from fgsim.models.gcn.treegcn import TreeGCN


class ModelClass(nn.Module):
    def __init__(self, features, degrees, support):
        self.batch_size = conf.loader.batch_size
        self.layer_num = len(features) - 1
        assert self.layer_num == len(
            degrees
        ), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super().__init__()

        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num - 1:
                self.gcn.add_module(
                    "TreeGCN_" + str(inx),
                    TreeGCN(
                        self.batch_size,
                        inx,
                        features,
                        degrees,
                        support=support,
                        node=vertex_num,
                        upsample=True,
                        activation=False,
                    ),
                )
            else:
                self.gcn.add_module(
                    "TreeGCN_" + str(inx),
                    TreeGCN(
                        self.batch_size,
                        inx,
                        features,
                        degrees,
                        support=support,
                        node=vertex_num,
                        upsample=True,
                        activation=True,
                    ),
                )
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        feat = self.gcn(tree)

        self.pointcloud = feat[-1]

        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]