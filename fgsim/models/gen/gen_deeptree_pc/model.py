from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import Data

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
        n_events: int,
        n_hidden_features: int,
        n_global: int,
        n_branches: int,
        n_levels: int,
        conv_name: str,
        conv_during_branching: bool,
        conv_parem,
        post_gen_mp_steps: int,
        conv_indp: bool = False,
        branching_indp: bool = False,
    ):
        super().__init__()
        self.n_events = n_events
        self.n_hidden_features = n_hidden_features
        self.n_features = conf.loader.n_features + n_hidden_features
        self.n_levels = n_levels
        self.n_global = n_global
        self.post_gen_mp_steps = post_gen_mp_steps
        self.convname = conv_name
        self.conv_during_branching = conv_during_branching
        self.conv_indp = conv_indp
        self.branching_indp = branching_indp

        # Calculate the output points
        self.output_points = sum([n_branches ** i for i in range(self.n_levels)])
        logger.debug(f"Generator output will be {self.output_points}")
        if conf.loader.max_points > self.output_points:
            raise RuntimeError(
                "Model cannot generate a sufficent number of points: "
                f"{conf.loader.max_points} < {self.output_points}"
            )
        conf.models.gen.output_points = self.output_points

        self.dyn_hlvs_layer = DynHLVsLayer(
            pre_nn=dnn_gen(self.n_features, self.n_features, n_layers=4).to(device),
            post_nn=dnn_gen(self.n_features * 2, self.n_global, n_layers=4).to(
                device
            ),
            n_events=n_events,
        )

        self.tree = Tree(
            n_events=self.n_events,
            n_features=self.n_features,
            n_branches=n_branches,
            n_levels=n_levels,
            device=device,
        )

        def gen_branching_layer():
            return BranchingLayer(
                tree=self.tree,
                proj_nn=dnn_gen(
                    self.n_features + n_global,
                    self.n_features * n_branches,
                    n_layers=4,
                ).to(device),
            )

        if self.branching_indp:
            self.branching_layers = [
                gen_branching_layer() for _ in range(self.n_levels - 1)
            ]
        else:
            branching_layer = gen_branching_layer()
            self.branching_layers = [
                branching_layer for _ in range(self.n_levels - 1)
            ]

        def gen_conv_layer():
            if self.convname == "GINConv":
                from torch_geometric.nn.conv import GINConv

                conv = GINConv(
                    dnn_gen(
                        self.n_features + n_global, self.n_features, n_layers=4
                    ).to(device)
                )
            elif self.convname == "AncestorConv":
                conv = AncestorConv(
                    n_features=self.n_features,
                    n_global=n_global,
                    **conv_parem,
                ).to(device)
            else:
                raise ImportError
            return conv

        if self.conv_indp:
            self.conv_layers = [gen_conv_layer() for _ in range(self.n_levels - 1)]
        else:
            conv_layer = gen_conv_layer()
            self.conv_layers = [conv_layer for _ in range(self.n_levels - 1)]

        if self.conv_indp:
            self.pp_conv_layers = [
                gen_conv_layer() for _ in range(self.post_gen_mp_steps)
            ]
        else:
            conv_layer = gen_conv_layer()
            self.pp_conv_layers = [
                conv_layer for _ in range(self.post_gen_mp_steps)
            ]

    def wrap_conv(
        self, graph: Data, global_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if self.convname == "GINConv":
            return {
                "x": torch.hstack([graph.x, global_features[graph.event]]),
                "edge_index": graph.edge_index,
            }

        elif self.convname == "AncestorConv":
            return {
                "x": graph.x,
                "edge_index": graph.edge_index,
                "edge_attr": graph.edge_attr,
                "event": graph.event,
                "global_features": global_features,
            }

    # Random vector to pc
    def forward(self, random_vector: torch.Tensor) -> Batch:
        n_events = self.n_events
        n_features = self.n_features

        graph = Data(
            x=random_vector.reshape(n_events, n_features),
            edge_index=torch.tensor([[], []], dtype=torch.long, device=device),
            edge_attr=torch.tensor([], dtype=torch.long, device=device),
            event=torch.arange(n_events, dtype=torch.long, device=device),
            tree=[[Node(torch.arange(n_events, dtype=torch.long, device=device))]],
        )

        for isplit in range(self.n_levels - 1):
            global_features = self.dyn_hlvs_layer(graph)
            graph = self.branching_layers[isplit](graph, global_features)
            if self.conv_during_branching:
                graph.x = self.conv_layers[isplit](
                    **(
                        self.wrap_conv(
                            graph,
                            global_features,
                        )
                    )
                )

        for ippstep in range(self.post_gen_mp_steps):
            global_features = self.dyn_hlvs_layer(graph)
            graph.x = self.pp_conv_layers[ippstep](
                **(
                    self.wrap_conv(
                        graph,
                        global_features,
                    )
                )
            )
        # slice the output the the corrent number of dimesions and create the batch
        batch = batch_tools.batch_from_pcs_list(
            graph.x[..., : conf.loader.n_features], graph.event
        )

        return batch
