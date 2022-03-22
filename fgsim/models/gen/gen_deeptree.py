from math import prod
from typing import Dict, List

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv import EdgeConv

from fgsim.config import conf, device
from fgsim.io.sel_seq import Batch, batch_tools
from fgsim.models.branching.branching import BranchingLayer, Tree
from fgsim.models.branching.node import Node
from fgsim.models.dnn_gen import dnn_gen
from fgsim.models.layer.ancestor_conv import AncestorConv
from fgsim.models.pooling.dyn_hlvs import DynHLVsLayer
from fgsim.monitoring.logger import logger


class ModelClass(nn.Module):
    def __init__(
        self,
        features: List[int],
        branches: List[int],
        n_global: int,
        conv_parem: Dict,
        branching_param: Dict,
        conv_name: str = "AncestorConv",
        conv_during_branching: bool = True,
        all_points: bool = False,
        pp_conv: bool = False,
    ):
        super().__init__()
        self.features = features
        self.branches = branches
        self.n_global = n_global
        self.conv_name = conv_name
        self.conv_during_branching = conv_during_branching
        self.all_points = all_points
        self.n_events = conf.loader.batch_size
        self.z_shape = conf.loader.batch_size, 1, self.features[0]
        self.pp_conv = pp_conv

        levels = len(self.branches)
        assert levels == len(features)
        assert branches[0] == 1

        # Calculate the output points
        if all_points:
            # If we use all point, we need to sum all of the splits
            self.output_points = sum(
                [prod(self.features[: i + 1]) for i in range(len(self.features))]
            )
        else:
            # otherwise the branches ^ n_splits works
            self.output_points = prod(self.features)
        logger.debug(f"Generator output will be {self.output_points}")
        if conf.loader.max_points > self.output_points:
            raise RuntimeError(
                "Model cannot generate a sufficent number of points: "
                f"{conf.loader.max_points} < {self.output_points}"
            )
        conf.models.gen.output_points = self.output_points

        self.tree = Tree(
            n_events=self.n_events,
            branches=self.branches,
            device=device,
        )
        self.dyn_hlvs_layers = nn.ModuleList(
            [
                DynHLVsLayer(
                    n_features=n_features,
                    n_global=n_global,
                    device=device,
                    n_events=self.n_events,
                )
                for n_features in features
            ]
        )

        self.branching_layers = nn.ModuleList(
            [
                BranchingLayer(
                    tree=self.tree,
                    level=level,
                    n_features=features[level - 1],
                    n_global=n_global,
                    **branching_param,
                )
                for level in range(1, levels)
            ]
        )

        def gen_conv_layer(level):
            if self.conv_name == "GINConv":
                from torch_geometric.nn.conv import GINConv

                conv = GINConv(
                    dnn_gen(
                        self.features[level - 1] + n_global, self.features[level]
                    ).to(device)
                )
            elif self.conv_name == "AncestorConv":
                conv = AncestorConv(
                    in_features=self.features[level - 1],
                    out_features=self.features[level],
                    n_global=n_global,
                    **conv_parem,
                ).to(device)
            else:
                raise ImportError
            return conv

        self.conv_layers = nn.ModuleList(
            [gen_conv_layer(level) for level in range(1, levels)]
        )

        if pp_conv:
            self.pp_convs = nn.ModuleList(
                [
                    EdgeConv(
                        nn=dnn_gen(self.features[-1] * 2, self.features[-1]),
                        aggr="add",
                    )
                    for _ in range(3)
                ]
            )

    def wrap_conv(self, graph: Data) -> Dict[str, torch.Tensor]:
        if self.conv_name == "GINConv":
            return {
                "x": torch.hstack([graph.x, graph.global_features[graph.event]]),
                "edge_index": graph.edge_index,
            }

        elif self.conv_name == "AncestorConv":
            return {
                "x": graph.x,
                "edge_index": graph.edge_index,
                "edge_attr": graph.edge_attr,
                "event": graph.event,
                "global_features": graph.global_features,
            }
        else:
            raise Exception

    # Random vector to pc
    def forward(self, random_vector: torch.Tensor) -> Batch:
        n_events = self.n_events
        n_global = self.n_global
        features = self.features
        branches = self.branches
        levels = len(branches)

        # Init the graph object
        graph = Data(
            x=random_vector.reshape(n_events, features[0]),
            edge_index=torch.empty(2, 0, dtype=torch.long, device=device),
            edge_attr=torch.empty(0, 1, dtype=torch.float, device=device),
            global_features=torch.empty(
                n_events, n_global, dtype=torch.float, device=device
            ),
            event=torch.arange(n_events, dtype=torch.long, device=device),
            tree=[[Node(torch.arange(n_events, dtype=torch.long, device=device))]],
        )

        # Do the branching
        for level in range(levels - 1):
            graph.global_features = self.dyn_hlvs_layers[level](graph)
            graph = self.branching_layers[level](graph)
            if self.conv_during_branching:
                graph.x = self.conv_layers[level](**(self.wrap_conv(graph)))

        # Edge_conv
        ei = knn_graph(x=graph.x, k=25, batch=graph.batch)
        if self.pp_conv:
            for conv in self.pp_convs:
                graph.x = conv(x=graph.x, edge_index=ei)

        # slice the output the the corrent number of dimesions and create the batch
        if self.all_points:
            batch = batch_tools.batch_from_pcs_list(
                graph.x[..., : conf.loader.n_features], graph.event
            )
        else:
            # self.output_points == n_branches ** (self.n_levels - 1)
            n_points_last_layer = self.output_points * conf.loader.batch_size
            # cut out the points generated by the last layer
            batch = batch_tools.batch_from_pcs_list(
                graph.x[-n_points_last_layer:, ..., : conf.loader.n_features],
                graph.event[-n_points_last_layer:],
            )

        return batch
