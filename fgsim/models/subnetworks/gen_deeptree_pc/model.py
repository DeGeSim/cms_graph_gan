import torch
import torch.nn as nn

from fgsim.config import conf, device
from fgsim.monitoring.logger import logger
from fgsim.types import Batch, Graph

from .ancestor_conv import AncestorConv
from .branching import BranchingLayer
from .dyn_hlvs import DynHLVsLayer
from .tree import Node


def dnn_gen(input_dim: int, output_dim: int, n_layers: int):
    if n_layers == 1:
        layers = [nn.Linear(input_dim, output_dim)]
    elif n_layers == 2:
        layers = [nn.Linear(input_dim, input_dim), nn.Linear(input_dim, output_dim)]
    else:
        layers = [
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, output_dim),
        ] + [nn.Linear(output_dim, output_dim) for _ in range(n_layers - 1)]
    seq = []
    for e in layers:
        seq.append(e)
        seq.append(nn.LeakyReLU(0.2))
    return nn.Sequential(*seq)


class ModelClass(nn.Module):
    def __init__(
        self,
        n_events: int,
        n_hidden_features: int,
        n_global: int,
        n_branches: int,
        n_levels: int,
        conv_name: str,
        conv_parem,
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
        logger.debug(f"Generator output will be {self.output_points}")
        if conf.loader.max_points > self.output_points:
            raise RuntimeError(
                "Model cannot generate a sufficent number of points: "
                f"{conf.loader.max_points} < {self.output_points}"
            )
        conf.models.gen.output_points = self.output_points

        self.dyn_hlvs_layer = DynHLVsLayer(
            pre_nn=dnn_gen(self.n_features, self.n_features, n_layers=4),
            post_nn=dnn_gen(self.n_features * 2, self.n_global, n_layers=4),
            n_events=n_events,
        )
        self.branching_layer = BranchingLayer(
            n_events=self.n_events,
            n_features=self.n_features,
            n_branches=n_branches,
            n_levels=n_levels,
            proj_nn=dnn_gen(
                self.n_features + n_global, self.n_features * n_branches, n_layers=4
            ),
            device=device,
        )

        if self.convname == "GINConv":
            from torch_geometric.nn.conv import GINConv

            self._conv = GINConv(
                dnn_gen(self.n_features + n_global, self.n_features, n_layers=4)
            )
        elif self.convname == "AncestorConv":
            self._conv = AncestorConv(
                msg_gen=dnn_gen(
                    self.n_features + self.n_global + 1, self.n_features, n_layers=4
                ),
                update_nn=dnn_gen(
                    2 * self.n_features + n_global, self.n_features, n_layers=4
                ),
                **conv_parem,
            )
        else:
            raise ImportError

    def conv(self, graph: Graph, global_features: torch.Tensor):
        if self.convname == "GINConv":
            return self._conv(
                x=torch.hstack([graph.x, global_features[graph.event]]),
                edge_index=graph.edge_index,
            )

        elif self.convname == "AncestorConv":
            return self._conv(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                event=graph.event,
                global_features=global_features,
            )

    # Random vector to pc
    def forward(self, random_vector: torch.Tensor) -> Batch:
        n_events = self.n_events
        n_features = self.n_features

        graph = Graph(
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

        batch = Batch.from_pcs_list(
            graph.x[:, : conf.loader.n_features], graph.event
        )
        return batch
