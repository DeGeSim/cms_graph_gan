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
        self,
        n_events: int,
        n_hidden_features: int,
        n_global: int,
        n_branches: int,
        n_splits: int,
    ):
        super().__init__()
        self.n_events = n_events
        self.n_hidden_features = n_hidden_features
        self.n_features = conf.loader.n_features + n_hidden_features
        self.n_splits = n_splits
        self.n_global = n_global
        self.output_points = sum(
            [n_branches ** i for i in range(self.n_splits + 1)]
        )
        logger.warning(f"Generator output will be {self.output_points}")
        if conf.loader.n_points < self.output_points:
            raise RuntimeError(
                "Event hast more points then the padding: "
                f"{conf.loader.n_points} < {self.output_points}"
            )

        self.global_aggr = GlobalDeepAggr(
            pre_nn=nn.Sequential(
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, n_global),
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
            n_events=self.n_events,
            n_features=self.n_features,
            n_branches=n_branches,
            proj_nn=nn.Sequential(
                nn.Linear(self.n_features + n_global, self.n_features * n_branches),
                nn.ReLU(),
                nn.Linear(
                    self.n_features * n_branches,
                    self.n_features * n_branches,
                ),
                nn.ReLU(),
                nn.Linear(
                    self.n_features * n_branches,
                    self.n_features * n_branches,
                ),
                nn.ReLU(),
            ),
        )
        self.ancester_conv = AncesterConv(
            msg_gen=nn.Sequential(
                nn.Linear(self.n_features + n_global, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
            ),
            update_nn=nn.Sequential(
                # agreegated features + previous feature vector + global
                nn.Linear(
                    2 * self.n_features + n_global, 2 * self.n_features + n_global
                ),
                nn.ReLU(),
                nn.Linear(2 * self.n_features + n_global, self.n_features),
                nn.ReLU(),
                nn.Linear(
                    self.n_features,
                    self.n_features,
                ),
                nn.ReLU(),
            ),
        )

    # Random vector to pc
    def forward(self, random_vector: torch.Tensor) -> torch.Tensor:
        n_events = self.n_events
        n_features = self.n_features

        graph = Data(
            x=random_vector.reshape(n_events, n_features),
            edge_index=torch.tensor([[], []], dtype=torch.long, device=device),
            edge_attr=torch.tensor([], dtype=torch.long, device=device),
            event=torch.arange(n_events, dtype=torch.long, device=device),
            tree=[[Node(torch.arange(n_events, dtype=torch.long, device=device))]],
        )

        for inx in range(self.n_splits):
            global_features = self.global_aggr(graph)
            graph = self.branching_nn(graph, global_features)
            graph.x = self.ancester_conv(graph, global_features)

        # No arange the events separatly by sorting the
        # arguments of the graph.event vector
        # Also only cut out the first conf.loader.n_features
        # features that have a meaning
        eidxs = torch.argsort(graph.event)
        # TODO write test
        # graph.event[torch.argsort(graph.event)[:10]]
        # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:3')
        #
        pcs = graph.x[eidxs, : conf.loader.n_features].reshape(
            n_events, -1, conf.loader.n_features
        )
        assert self.output_points == pcs.shape[1]

        pcs_padded = torch.nn.functional.pad(
            pcs,
            (0, 0, 0, conf.loader.n_points - pcs.shape[1], 0, 0),
            mode="constant",
            value=0,
        )
        return pcs_padded
