import torch
import torch.nn as nn
from torch_geometric.data import Data

from fgsim.config import conf, device
from fgsim.monitoring.logger import logger

from .ancestor_conv import AncestorConvLayer
from .branching import BranchingLayer
from .dyn_hlvs import DynHLVsLayer
from .tree import Node


class ModelClass(nn.Module):
    def __init__(
        self,
        n_events: int,
        n_hidden_features: int,
        n_global: int,
        n_branches: int,
        n_levels: int,
        conv_name: str,
        post_gen_mp_steps: int,
    ):
        super().__init__()
        self.n_events = n_events
        self.n_hidden_features = n_hidden_features
        self.n_features = conf.loader.n_features + n_hidden_features
        self.n_levels = n_levels
        self.n_global = n_global
        self.post_gen_mp_steps = post_gen_mp_steps
        self.convname = conv_name
        self.output_points = sum([n_branches ** i for i in range(self.n_levels)])
        logger.warning(f"Generator output will be {self.output_points}")
        if conf.loader.max_points > self.output_points:
            raise RuntimeError(
                "Model cannot generate a sufficent number of points: "
                f"{conf.loader.max_points} < {self.output_points}"
            )

        self.dyn_hlvs_layer = DynHLVsLayer(
            pre_nn=nn.Sequential(
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, n_global),
                nn.ReLU(),
            ),
            post_nn=nn.Sequential(
                nn.Linear(n_global * 2, n_global * 2),
                nn.ReLU(),
                nn.Linear(n_global * 2, n_global * 2),
                nn.ReLU(),
                nn.Linear(n_global * 2, n_global),
                nn.ReLU(),
            ),
            n_events=n_events,
        )
        self.branching_layer = BranchingLayer(
            n_events=self.n_events,
            n_features=self.n_features,
            n_branches=n_branches,
            n_levels=n_levels,
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
            device=device,
        )

        if self.convname == "GINConv":
            from torch_geometric.nn.conv import GINConv

            self._conv = GINConv(
                nn.Sequential(
                    nn.Linear(
                        self.n_features + n_global, self.n_features + n_global
                    ),
                    nn.ReLU(),
                    nn.Linear(self.n_features + n_global, self.n_features),
                    nn.ReLU(),
                    nn.Linear(self.n_features, self.n_features),
                    nn.ReLU(),
                    nn.Linear(self.n_features, self.n_features),
                    nn.ReLU(),
                )
            )
        elif self.convname == "AncestorConv":
            self._conv = AncestorConvLayer(
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
                        2 * self.n_features + n_global,
                        2 * self.n_features + n_global,
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
        else:
            raise ImportError

    def conv(self, graph: Data, global_features: torch.Tensor):
        if self.convname == "GINConv":
            return self._conv(
                x=torch.hstack([graph.x, global_features[graph.event]]),
                edge_index=graph.edge_index,
            )

        elif self.convname == "AncestorConv":
            return self._conv(
                x=graph.x,
                edge_index=graph.edge_index,
                event=graph.event,
                global_features=global_features,
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

        for _ in range(self.n_levels - 1):
            global_features = self.dyn_hlvs_layer(graph)
            graph = self.branching_layer(graph, global_features)
            graph.x = self.conv(
                graph,
                global_features,
            )

        for _ in range(self.post_gen_mp_steps):
            global_features = self.dyn_hlvs_layer(graph)
            graph.x = self.conv(
                graph,
                global_features,
            )

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

        return pcs
