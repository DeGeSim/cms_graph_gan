from math import prod
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from fgsim.config import conf, device
from fgsim.models.common import DynHLVsLayer
from fgsim.models.common.deeptree import (
    BranchingLayer,
    DeepConv,
    GraphTreeWrapper,
    MPLSeq,
    Tree,
    TreeGenType,
    graph_tree_to_batch,
)
from fgsim.monitoring.logger import logger


class ModelClass(nn.Module):
    def __init__(
        self,
        n_global: int,
        conv_parem: Dict,
        child_param: Dict,
        branching_param: Dict,
        all_points: bool = False,
    ):
        super().__init__()
        self.n_global = n_global
        self.all_points = all_points
        self.batch_size = conf.loader.batch_size

        self.features = conf.tree.features
        self.branches = conf.tree.branches
        n_levels = len(self.features)

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
        if conf.loader.n_points > self.output_points:
            raise RuntimeError(
                "Model cannot generate a sufficent number of points: "
                f"{conf.loader.n_points} < {self.output_points}"
            )
        conf.models.gen.output_points = self.output_points

        self.tree = Tree(
            batch_size=conf.loader.batch_size,
            branches=OmegaConf.to_container(conf.tree.branches),
            features=OmegaConf.to_container(conf.tree.features),
            device=device,
        )

        if n_global > 0:
            self.dyn_hlvs_layers = nn.ModuleList(
                [
                    DynHLVsLayer(
                        n_features=self.features[-1],
                        n_global=n_global,
                        device=device,
                        batch_size=self.batch_size,
                    )
                    for _ in self.features
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
                for level in range(n_levels - 1)
            ]
        )

        self.ancestor_conv_layers = nn.ModuleList(
            [
                DeepConv(
                    in_features=self.features[level],
                    out_features=self.features[level + 1],
                    n_global=n_global,
                    **conv_parem,
                )
                for level in range(n_levels - 1)
            ]
        )
        self.child_conv_layers = nn.ModuleList(
            [
                MPLSeq(
                    in_features=self.features[level],
                    out_features=self.features[level],
                    n_mpl=child_param["n_mpl"],
                    n_hidden_nodes=max(
                        child_param["n_hidden_nodes"], self.features[level]
                    ),
                    **conv_parem,
                )
                for level in range(1, n_levels)
            ]
        )

    def forward(self, random_vector: torch.Tensor) -> Batch:
        batch_size = self.batch_size
        features = self.features
        branches = self.branches
        n_levels = len(branches)

        # Init the graph object
        graph_tree = GraphTreeWrapper(
            TreeGenType(
                tftx=random_vector.reshape(batch_size, features[0]),
                batch_size=batch_size,
            )
        )

        # Do the branching
        for ilevel in range(n_levels):
            # Assign the global features
            if self.n_global > 0:
                graph_tree.global_features = self.dyn_hlvs_layers[ilevel](
                    graph_tree.tftx_by_level[ilevel][..., : features[-1]],
                    graph_tree.tbatch[graph_tree.idxs_by_level[ilevel]],
                )
            else:
                graph_tree.global_features = torch.empty(
                    batch_size, 0, dtype=torch.float, device=graph_tree.tftx.device
                )

            graph_tree = self.branching_layers[ilevel](graph_tree)

            graph_tree.tftx = self.ancestor_conv_layers[ilevel](
                tftx=graph_tree.tftx,
                edge_index=self.tree.ancestor_ei(ilevel),
                edge_attr=self.tree.ancestor_ea(ilevel),
                tbatch=graph_tree.tbatch,
                global_features=graph_tree.global_features,
            )
            graph_tree.tftx = self.child_conv_layers[ilevel](
                tftx=graph_tree.tftx,
                edge_index=self.tree.children_ei(ilevel),
                tbatch=graph_tree.tbatch,
                global_features=graph_tree.global_features,
            )
        graph_tree.data.x = graph_tree.tftx_by_level[-1]
        graph_tree.data.batch = graph_tree.batch_by_level[-1]

        return graph_tree_to_batch(graph_tree)
