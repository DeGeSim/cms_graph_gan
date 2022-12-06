from math import prod
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.models.common import DynHLVsLayer, FtxScaleLayer, MPLSeq
from fgsim.models.common.deeptree import (
    BranchingLayer,
    GraphTreeWrapper,
    Tree,
    TreeGenType,
)
from fgsim.monitoring.logger import logger

# from fgsim.plot.model_plotter import model_plotter


class ModelClass(nn.Module):
    def __init__(
        self,
        n_global: int,
        n_cond: int,
        ancestor_mpl: Dict,
        child_mpl: Dict,
        branching_param: Dict,
        all_points: bool,
        final_layer_scaler: bool,
        connect_all_ancestors: bool,
        dim_red_in_branching: bool,
        **kwargs,
    ):
        super().__init__()
        self.n_global = n_global
        self.n_cond = n_cond
        self.all_points = all_points
        self.batch_size = conf.loader.batch_size
        self.final_layer_scaler = final_layer_scaler
        self.ancestor_mpl = ancestor_mpl
        self.child_mpl = child_mpl
        self.dim_red_in_branching = dim_red_in_branching

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
                    dim_red=self.dim_red_in_branching,
                    **branching_param,
                )
                for level in range(n_levels - 1)
            ]
        )

        self.ancestor_conv_layers = nn.ModuleList(
            [
                self.wrap_layer_init(ilevel, type="ac")
                for ilevel in range(n_levels - 1)
            ]
        )

        self.child_conv_layers = nn.ModuleList(
            [
                self.wrap_layer_init(ilevel, type="child")
                for ilevel in range(n_levels - 1)
            ]
        )

        if self.final_layer_scaler:
            self.ftx_scaling = FtxScaleLayer(self.features[-1])

        # Allocate the Tensors used later to construct the batch
        self.presaved_batch: Optional[Batch] = None
        self.presaved_batch_indexing: Optional[torch.Tensor] = None

    def wrap_layer_init(self, ilevel, type: str):
        if type == "ac":
            conv_param = self.ancestor_mpl
        elif type == "child":
            conv_param = self.child_mpl
        else:
            raise Exception

        return MPLSeq(
            in_features=self.features[ilevel + int(self.dim_red_in_branching)]
            if type == "ac"
            else self.features[ilevel + 1],
            out_features=self.features[ilevel + 1],
            n_cond=self.n_cond,
            n_global=self.n_global,
            **conv_param,
        )

    def forward(self, random_vector: torch.Tensor, cond: torch.Tensor) -> Batch:
        batch_size = self.batch_size
        features = self.features
        n_levels = len(self.features)

        # Init the graph object
        graph_tree: GraphTreeWrapper = GraphTreeWrapper(
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
        # model_plotter.save_tensor("level0", graph_tree.tftx_by_level[0])

        # Do the branching
        for ilevel in range(n_levels - 1):
            assert graph_tree.tftx.shape[1] == self.tree.features[ilevel]
            assert graph_tree.tftx.shape[0] == (
                self.tree.tree_lists[ilevel][-1].idxs[-1] + 1
            )
            # Assign the global features
            graph_tree.global_features = self.dyn_hlvs_layers[ilevel](
                x=graph_tree.tftx_by_level[ilevel][..., : features[-1]],
                cond=graph_tree.cond,
                batch=graph_tree.tbatch[graph_tree.idxs_by_level[ilevel]],
            )

            graph_tree = self.branching_layers[ilevel](graph_tree)
            assert (
                graph_tree.tftx.shape[1]
                == self.tree.features[ilevel + int(self.dim_red_in_branching)]
            )
            assert graph_tree.tftx.shape[0] == (
                self.tree.tree_lists[ilevel + 1][-1].idxs[-1] + 1
            )
            # model_plotter.save_tensor(
            #     f"branching output level{ilevel+1}", graph_tree.tftx_by_level[-1]
            # )

            graph_tree.tftx = self.ancestor_conv_layers[ilevel](
                x=graph_tree.tftx,
                cond=graph_tree.cond,
                edge_index=self.tree.ancestor_ei(ilevel + 1),
                edge_attr=self.tree.ancestor_ea(ilevel + 1),
                batch=graph_tree.tbatch,
                global_features=graph_tree.global_features,
            )
            assert graph_tree.tftx.shape[1] == self.tree.features[ilevel + 1]
            assert graph_tree.tftx.shape[0] == (
                self.tree.tree_lists[ilevel + 1][-1].idxs[-1] + 1
            )
            # if len(self.ancestor_conv_layers) > 0:
            #     model_plotter.save_tensor(
            #         f"ancestor conv output level{ilevel+1}",
            #         graph_tree.tftx_by_level[-1],
            #     )

            graph_tree.tftx = self.child_conv_layers[ilevel](
                x=graph_tree.tftx,
                cond=graph_tree.cond,
                edge_index=self.tree.children_ei(ilevel + 1),
                edge_attr=None,
                batch=graph_tree.tbatch,
                global_features=graph_tree.global_features,
            )
            assert graph_tree.tftx.shape[1] == self.tree.features[ilevel + 1]
            assert graph_tree.tftx.shape[0] == (
                self.tree.tree_lists[ilevel + 1][-1].idxs[-1] + 1
            )

            # if len(self.child_conv_layers) > 0:
            #     model_plotter.save_tensor(
            #         f"child conv output level{ilevel+1}",
            #         graph_tree.tftx_by_level[-1],
            #     )

        if self.presaved_batch is None:
            (
                self.presaved_batch,
                self.presaved_batch_indexing,
            ) = graph_tree.get_batch_skeleton()

        batch = self.presaved_batch.clone()
        batch.x = graph_tree.tftx_by_level[-1][self.presaved_batch_indexing]
        if self.final_layer_scaler:
            batch.x = self.ftx_scaling(batch.x)

        assert batch.x.shape[0] == conf.loader.n_points * conf.loader.batch_size
        assert batch.x.shape[-1] == conf.loader.n_features
        assert batch.num_graphs == conf.loader.batch_size
        return batch

    def to(self, device):
        super().to(device)
        self.tree.to(device)
        return self
