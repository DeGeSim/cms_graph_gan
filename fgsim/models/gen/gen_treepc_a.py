import torch.nn as nn
from torch_geometric.data import Batch, Data

from fgsim.config import conf
from fgsim.models.branching.treegcn_a import TreeGCN


class ModelClass(nn.Module):
    def __init__(self, features, degrees, support):
        self.batch_size = conf.loader.batch_size

        self.z_shape = conf.loader.batch_size, 1, features[0]

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
                        batch=self.batch_size,
                        depth=inx,
                        features=features,
                        degrees=degrees,
                        support=support,
                        n_parents=vertex_num,
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
                        n_parents=vertex_num,
                        activation=True,
                    ),
                )
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, z):
        tree = [z]
        feat = self.gcn(tree)

        self.pointcloud = feat[-1]
        batch = Batch.from_data_list([Data(x=points) for points in self.pointcloud])
        return batch

    def getPointcloud(self):
        return self.pointcloud[-1]
