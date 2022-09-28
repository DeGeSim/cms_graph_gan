from math import prod
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.models.common import DynHLVsLayer
from fgsim.models.common.deeptree import (
    BranchingLayer,
    GraphTreeWrapper,
    Tree,
    TreeGenType,
)
from fgsim.models.common.deeptree.ftxscale import FtxScaleLayer
from fgsim.models.common.ffn import FFN
from fgsim.monitoring.logger import logger


class ModelClass(nn.Module):
    def __init__(
        self,
        n_global: int,
        n_cond: int,
        conv_name: str,
        conv_param: Dict,
        branching_param: Dict,
        all_points: bool,
        final_layer_scaler: bool,
        connect_all_ancestors: bool,
        **kwargs,
    ):
        super().__init__()
        self.n_global = n_global
        self.n_cond = n_cond
        self.all_points = all_points
        self.batch_size = conf.loader.batch_size
        self.final_layer_scaler = final_layer_scaler
        self.conv_name = conv_name
        self.conv_param = conv_param

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
            connect_all_ancestors=connect_all_ancestors,
            branches=OmegaConf.to_container(conf.tree.branches),
            features=OmegaConf.to_container(conf.tree.features),
        )

        self.dyn_hlvs_layers = nn.ModuleList(
            [
                DynHLVsLayer(
                    n_features=self.features[-1],
                    n_cond=self.n_cond,
                    n_global=n_global,
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
                    n_cond=self.n_cond,
                    **branching_param,
                )
                for level in range(n_levels - 1)
            ]
        )

        self.ancestor_conv_layers = nn.ModuleList(
            [self.wrap_layer_init(ilevel) for ilevel in range(n_levels - 1)]
        )
        # self.child_conv_layers = nn.ModuleList(
        #     [
        #         MPLSeq(
        #             in_features=self.features[level],
        #             out_features=self.features[level],
        #             n_mpl=child_param["n_mpl"],
        #             n_hidden_nodes=max(
        #                 child_param["n_hidden_nodes"], self.features[level]
        #             ),
        #             n_global=n_global,
        #             n_cond=self.n_cond,
        #             **conv_param,
        #         )
        #         for level in range(1, n_levels)
        #     ]
        # )
        if self.final_layer_scaler:
            self.ftx_scaling = FtxScaleLayer(self.features[-1])

    def wrap_layer_init(self, ilevel):
        if self.conv_name == "gincconv":
            from fgsim.models.common import GINCConv

            return GINCConv(
                FFN(
                    self.features[ilevel] + self.n_cond + self.n_global,
                    self.features[ilevel + 1],
                )
            )
        elif self.conv_name == "ginconv":
            from torch_geometric.nn import GINConv

            return GINConv(
                FFN(
                    self.features[ilevel] + self.n_cond + self.n_global,
                    self.features[ilevel + 1],
                )
            )
        elif self.conv_name == "deepconv":
            from fgsim.models.common.deeptree import DeepConv

            return DeepConv(
                in_features=self.features[ilevel],
                out_features=self.features[ilevel + 1],
                n_global=self.n_global,
                n_cond=self.n_cond,
                **self.conv_param,
            )

    def wrap_ac(
        self, *, layer, x, cond, edge_index, edge_attr, batch, global_features
    ):
        if self.conv_name == "gincconv":
            return layer(
                x,
                torch.hstack(
                    (
                        cond[batch],
                        global_features[batch],
                    )
                ),
                edge_index,
            )
        elif self.conv_name == "ginconv":
            return layer(
                torch.hstack(
                    (
                        x,
                        cond[batch],
                        global_features[batch],
                    )
                ),
                edge_index,
            )
        elif self.conv_name == "deepconv":
            return layer(
                x=x,
                cond=cond,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch,
                global_features=global_features,
            )

    def forward(self, random_vector: torch.Tensor, cond: torch.Tensor) -> Batch:
        batch_size = self.batch_size
        features = self.features
        n_levels = len(self.features)

        # Init the graph object
        graph_tree = GraphTreeWrapper(
            TreeGenType(
                tftx=random_vector.reshape(batch_size, features[0]),
                batch_size=batch_size,
            )
        )
        # overwrite the first features of the reandom vector with the condition
        graph_tree.cond = (
            cond.clone()
            .detach()
            .reshape(batch_size, self.n_cond)
            .requires_grad_(True)
        )
        graph_tree.tftx_by_level[0][..., : cond.shape[-1]] = graph_tree.cond

        # Do the branching
        for ilevel in range(n_levels - 1):
            # Assign the global features
            graph_tree.global_features = self.dyn_hlvs_layers[ilevel](
                x=graph_tree.tftx_by_level[ilevel][..., : features[-1]],
                cond=graph_tree.cond,
                batch=graph_tree.tbatch[graph_tree.idxs_by_level[ilevel]],
            )

            graph_tree = self.branching_layers[ilevel](graph_tree)

            graph_tree.tftx = self.wrap_ac(
                layer=self.ancestor_conv_layers[ilevel],
                x=graph_tree.tftx,
                cond=graph_tree.cond,
                edge_index=self.tree.ancestor_ei(ilevel + 1),
                edge_attr=self.tree.ancestor_ea(ilevel + 1),
                batch=graph_tree.tbatch,
                global_features=graph_tree.global_features,
            )
            # graph_tree.tftx = self.child_conv_layers[ilevel](
            #     x=graph_tree.tftx,
            #     cond=graph_tree.cond,
            #     edge_index=self.tree.children_ei(ilevel+1),
            #     batch=graph_tree.tbatch,
            #     global_features=graph_tree.global_features,
            # )

        batch = graph_tree.to_batch()
        if self.final_layer_scaler:
            batch.x = self.ftx_scaling(batch.x)

        return batch

    def to(self, device):
        super().to(device)
        self.tree.to(device)
        return self
