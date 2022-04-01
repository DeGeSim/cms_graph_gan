from math import prod
from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import Data

from fgsim.config import conf, device
from fgsim.io.sel_seq import Batch, batch_tools
from fgsim.models.branching.branching import BranchingLayer, Tree
from fgsim.models.branching.node import Node
from fgsim.models.ffn import FFN
from fgsim.models.layer.ancestor_conv import AncestorConv
from fgsim.models.pooling.dyn_hlvs import DynHLVsLayer
from fgsim.monitoring.logger import logger

# from torch_geometric.nn import knn_graph
# from torch_geometric.nn.conv import EdgeConv


tree = Tree(
    batch_size=conf.loader.batch_size,
    branches=conf.tree.branches,
    features=conf.tree.features,
    device=device,
)


class ModelClass(nn.Module):
    def __init__(
        self,
        n_global: int,
        conv_parem: Dict,
        branching_param: Dict,
        conv_name: str = "AncestorConv",
        conv_during_branching: bool = True,
        all_points: bool = False,
        pp_conv: bool = False,
    ):
        super().__init__()
        self.n_global = n_global
        self.conv_name = conv_name
        self.conv_during_branching = conv_during_branching
        self.all_points = all_points
        self.batch_size = conf.loader.batch_size
        self.pp_conv = pp_conv

        self.features = conf.tree.features
        self.branches = conf.tree.branches
        levels = len(self.features)

        # Shape of the random vector
        self.z_shape = conf.loader.batch_size, 1, self.features[0]

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

        self.tree = tree

        self.dyn_hlvs_layers = nn.ModuleList(
            [
                DynHLVsLayer(
                    n_features=n_features,
                    n_global=n_global,
                    device=device,
                    batch_size=self.batch_size,
                )
                for n_features in self.features
            ]
        )

        self.branching_layers = nn.ModuleList(
            [
                BranchingLayer(
                    tree=self.tree,
                    level=level,
                    n_global=n_global,
                    **branching_param,
                )
                for level in range(levels - 1)
            ]
        )

        def gen_conv_layer(level):
            if self.conv_name == "GINConv":
                from torch_geometric.nn.conv import GINConv

                conv = GINConv(
                    FFN(
                        self.features[level] + n_global, self.features[level + 1]
                    ).to(device)
                )
            elif self.conv_name == "AncestorConv":
                conv = AncestorConv(
                    in_features=self.features[level],
                    out_features=self.features[level + 1],
                    n_global=n_global,
                    **conv_parem,
                ).to(device)
            else:
                raise ImportError
            return conv

        self.conv_layers = nn.ModuleList(
            [gen_conv_layer(level) for level in range(levels - 1)]
        )

        # if pp_conv:
        #     self.pp_convs = nn.ModuleList(
        #         [
        #             EdgeConv(
        #                 nn=FFN(self.features[-1] * 2, self.features[-1]),
        #                 aggr="add",
        #             )
        #             for _ in range(3)
        #         ]
        #     )

    def wrap_conv(self, graph: Data) -> Dict[str, torch.Tensor]:
        if self.conv_name == "GINConv":
            return {
                "x": torch.hstack([graph.x, graph.global_features[graph.batch]]),
                "edge_index": graph.edge_index,
            }

        elif self.conv_name == "AncestorConv":
            return {
                "x": graph.x,
                "edge_index": graph.edge_index,
                "edge_attr": graph.edge_attr,
                "batch": graph.batch,
                "global_features": graph.global_features,
            }
        else:
            raise Exception

    # Random vector to pc
    def forward(self, random_vector: torch.Tensor) -> Batch:
        batch_size = self.batch_size
        n_global = self.n_global
        features = self.features
        branches = self.branches
        levels = len(branches)

        # Init the graph object
        graph = Data(
            levels=[random_vector.reshape(batch_size, features[0])],
            children=[],
            edge_index=torch.empty(2, 0, dtype=torch.long, device=device),
            edge_attr=torch.empty(0, 1, dtype=torch.float, device=device),
            global_features=torch.empty(
                batch_size, n_global, dtype=torch.float, device=device
            ),
            batch=torch.arange(batch_size, dtype=torch.long, device=device),
            tree=[
                [Node(torch.arange(batch_size, dtype=torch.long, device=device))]
            ],
        )

        # Do the branching
        for ilevel in range(levels - 1):
            graph.global_features = self.dyn_hlvs_layers[ilevel](
                graph.levels[ilevel], graph.batch
            )
            graph = self.branching_layers[ilevel](graph)
            if self.conv_during_branching:
                graph.x = self.conv_layers[ilevel](**(self.wrap_conv(graph)))

        # # Edge_conv
        # if self.pp_conv:
        #     ei = knn_graph(x=graph.x, k=25, batch=graph.batch)
        #     for conv in self.pp_convs:
        #         graph.x = conv(x=graph.x, edge_index=ei)

        # slice the output the the corrent number of dimesions and create the batch
        if self.all_points:
            batch = batch_tools.batch_from_pcs_list(
                graph.x[..., : conf.loader.n_features], graph.batch
            )
        else:
            n_points_last_layer = self.output_points * conf.loader.batch_size
            # cut out the points generated by the last layer
            batch = batch_tools.batch_from_pcs_list(
                graph.x[-n_points_last_layer:, ..., : conf.loader.n_features],
                graph.batch[-n_points_last_layer:],
            )

        return batch
