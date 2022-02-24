from typing import Optional, Union

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from .dnn import dnn_gen


class AncestorConv(MessagePassing):
    """
    1. The global features are concatenated with the node
       features and passed to the message generation layer.
    2. The messages are aggregated and passed to the update layer.
    3. The update layer returns the updated node features."""

    def __init__(
        self,
        n_features: int,
        n_global: int,
        add_self_loops: bool,
        msg_nn: bool,
        msg_nn_include_edge_attr: bool,
        msg_nn_include_global: bool,
        upd_nn: bool,
        upd_nn_include_global: bool,
    ):
        super().__init__(aggr="add", flow="source_to_target")
        self.n_features = n_features
        self.n_global = n_global
        self.add_self_loops = add_self_loops
        self.msg_nn = msg_nn
        self.msg_nn_include_edge_attr = msg_nn_include_edge_attr
        self.msg_nn_include_global = msg_nn_include_global
        self.upd_nn = upd_nn
        self.upd_nn_include_global = upd_nn_include_global

        # MSG NN
        self.msg_gen: Union[torch.Module, torch.nn.Identity] = torch.nn.Identity()
        if msg_nn:
            self.msg_gen = dnn_gen(
                n_features
                + (n_global if msg_nn_include_global else 0)
                + (1 if msg_nn_include_edge_attr else 0),
                self.n_features,
                n_layers=4,
            )
        else:
            assert not (msg_nn_include_edge_attr or msg_nn_include_global)

        # UPD NN
        self.update_nn: Union[torch.Module, torch.nn.Identity] = torch.nn.Identity()
        if upd_nn:
            self.update_nn = dnn_gen(
                2 * n_features + (n_global if upd_nn_include_global else 0),
                n_features,
                n_layers=4,
            )
        else:
            assert not upd_nn_include_global

    def forward(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        event: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        global_features: Optional[torch.Tensor],
    ):
        if self.msg_nn_include_edge_attr:
            assert edge_attr is not None
        if self.msg_nn_include_global:
            assert global_features is not None
        num_nodes = x.size(0)
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index=edge_index,
                edge_attr=edge_attr,
                fill_value=0,
                num_nodes=num_nodes,
            )

        # Generate a global feature vector in shape of x
        glo_ftx_mtx = (
            global_features[event, :] if self.msg_nn_include_global else None
        )
        # If the egde_attrs are included, we transforming the message
        if self.msg_nn_include_edge_attr:
            new_x = self.propagate(
                edge_index=edge_index,
                edge_attr=edge_attr,
                x=x,
                glo_ftx_mtx=glo_ftx_mtx,
                size=(num_nodes, num_nodes),
            )
        # If the egde attr are not included, we apply a transformation
        # before the message instead of transforming the message
        else:
            # Generate a global feature vector in shape of x
            xtransform = self.msg_gen(
                torch.hstack(
                    [x] + ([glo_ftx_mtx] if self.msg_nn_include_global else [])
                )
            )
            new_x = self.propagate(
                edge_index=edge_index,
                edge_attr=torch.empty_like(edge_attr),  # required, pass as empty
                x=xtransform,
                glo_ftx_mtx=torch.empty_like(
                    glo_ftx_mtx
                ),  # required, pass as empty
                size=(num_nodes, num_nodes),
            )

        # self loop
        return new_x

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        glo_ftx_mtx_j: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Transform node feature matrix with the global features
        if self.msg_nn and self.msg_nn_include_edge_attr:
            xtransform = self.msg_gen(
                torch.hstack(
                    [x_j]
                    + ([glo_ftx_mtx_j] if self.msg_nn_include_global else [])
                    + (
                        [edge_attr.reshape(x_j.size(0), -1)]
                        if self.msg_nn_include_edge_attr
                        else []
                    )
                )
            )
        else:
            xtransform = x_j
        return xtransform

    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
        glo_ftx_mtx: torch.Tensor,
    ) -> torch.Tensor:
        if self.upd_nn:
            upd = self.update_nn(
                torch.hstack(
                    [x]
                    + ([glo_ftx_mtx] if self.upd_nn_include_global else [])
                    + [aggr_out]
                )
            )
        else:
            upd = aggr_out
        return upd
