import torch
import torch.nn as nn
from torch_geometric.data import Data

from fgsim.config import conf, device
from fgsim.monitoring.logger import logger

from .ancester_conv import AncesterConv
from .global_feedback import GlobalDeepAggr
from .splitting import NodeSpliter
from .tree import Node


class ModelClass(nn.Module):
    def __init__(
        self, n_hidden_features: int, n_global: int, n_branches: int, n_splits: int
    ):
        super().__init__()
        n_features = conf.loader.n_features + n_hidden_features
        self.n_splits = n_splits
        # Compute the number of nodes generated in the tree
        # Should be in the configuration ideally
        # Hopefully when this is implemented
        # https://github.com/omry/omegaconf/issues/91
        conf.training["n_points"] = sum(
            [n_branches ** i for i in range(self.n_splits)]
        )

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
            n_branches=n_branches,
            proj_nn=nn.Sequential(
                nn.Linear(n_features + n_global, n_features * n_branches),
                nn.ReLU(),
                nn.Linear(
                    n_features * n_branches,
                    n_features * n_branches,
                ),
                nn.ReLU(),
                nn.Linear(
                    n_features * n_branches,
                    n_features * n_branches,
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
                nn.Linear(2 * n_features + n_global, n_features),
                nn.ReLU(),
                nn.Linear(
                    n_features,
                    n_features,
                ),
                nn.ReLU(),
            ),
        )

    # Random vector to pc
    def forward(self, random_vector: torch.Tensor) -> torch.Tensor:
        pc_list = []
        for ipc, x in enumerate(random_vector):
            logger.info(f"PC #{ipc}")
            edge_index = torch.tensor([[], []], dtype=torch.long, device=device)
            graph = Data(x=x, edge_index=edge_index)
            graph.tree = [[Node(0)]]
            for inx in range(self.n_splits):
                global_features = self.global_aggr(graph)
                graph = self.branching_nn(graph, global_features)
                graph.x = self.ancester_conv(graph, global_features)
            pc_list.append(graph.x)
        return torch.vstack(pc_list)
