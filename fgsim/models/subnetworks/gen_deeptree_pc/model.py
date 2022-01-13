import torch.nn as nn

from fgsim.config import conf

from .ancester_conv import AncesterConv
from .global_feedback import GlobalDeepAggr
from .splitting import NodeSpliter
from .tree import Node


class ModelClass(nn.Module):
    def __init__(self):
        super().__init__()
        param = conf.models.gen.param
        # Compute the number of nodes generated in the tree
        conf.training["n_points"] = sum(
            [param["n_branches"] ** i for i in range(param["n_branches"])]
        )
        n_features = (
            conf.loader.n_features + conf.models.gen.param.n_hidden_features
        )
        n_global = param["n_global"]
        self.global_aggr = GlobalDeepAggr(
            pre_nn=nn.Sequential(
                nn.Linear(n_features, n_features),
                nn.ReLU(),
                nn.Linear(n_features, n_global),
                nn.ReLU(),
            ),
            post_nn=nn.Sequential(
                nn.Linear(n_global, n_global),
                nn.ReLU(),
                nn.Linear(n_global, n_global),
                nn.ReLU(),
            ),
        )
        self.branching_nn = NodeSpliter(
            n_features=n_features,
            n_branches=param["n_branches"],
            proj_nn=nn.Sequential(
                nn.Linear(n_features + n_global, n_features * param["n_branches"]),
                nn.ReLU(),
                nn.Linear(
                    n_features * param["n_branches"],
                    n_features * param["n_branches"],
                ),
                nn.ReLU(),
                nn.Linear(
                    n_features * param["n_branches"],
                    n_features * param["n_branches"],
                ),
                nn.ReLU(),
            ),
        )
        self.ancester_conv = AncesterConv(
            msg_gen=nn.Sequential(
                nn.Linear(n_features + n_global, n_features),
                nn.ReLU(),
                nn.Linear(n_features, n_features),
                nn.ReLU(),
                nn.Linear(n_features, n_features),
                nn.ReLU(),
            ),
            update_nn=nn.Sequential(
                # agreegated features + previous feature vector + global
                nn.Linear(2 * n_features + n_global, 2 * n_features + n_global),
                nn.ReLU(),
                nn.Linear(
                    2 * n_features + n_global, n_features * param["n_branches"]
                ),
                nn.ReLU(),
                nn.Linear(
                    n_features * param["n_branches"],
                    n_features * param["n_branches"],
                ),
                nn.ReLU(),
            ),
        )

    def forward(self, random_vector):
        root = Node(random_vector)
        tree_layers = [[root]]
        for inx in range(self.layer_num):
            global_features = self.global_aggr(root.get_full_tree())
            tree_layers = self.branching_nn(tree_layers[inx], global_features)
            self.ancester_conv(root, global_features)

        return self.pointcloud
