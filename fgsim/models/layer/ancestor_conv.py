from typing import Union

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from fgsim.models.dnn_gen import dnn_gen


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
        add_self_loops: bool = True,
        msg_nn_bool: bool = True,
        upd_nn_bool: bool = True,
        msg_nn_include_edge_attr: bool = False,
        msg_nn_include_global: bool = True,
        upd_nn_include_global: bool = True,
    ):

        super().__init__(aggr="add", flow="source_to_target")
        self.n_features = n_features
        self.n_global = n_global
        self.add_self_loops = add_self_loops
        self.msg_nn_bool = msg_nn_bool
        self.upd_nn_bool = upd_nn_bool
        self.msg_nn_include_edge_attr = msg_nn_include_edge_attr
        self.msg_nn_include_global = msg_nn_include_global
        self.upd_nn_include_global = upd_nn_include_global

        # MSG NN
        self.msg_nn: Union[torch.nn.Module, torch.nn.Identity] = torch.nn.Identity()
        if self.msg_nn_bool:
            self.msg_nn = dnn_gen(
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
        if upd_nn_bool:
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
        edge_attr: torch.Tensor = torch.tensor([]),
        global_features: torch.Tensor = torch.tensor([]),
    ):
        num_nodes = x.shape[0]

        num_edges = edge_index.shape[1]
        num_events = event[-1] + 1

        if len(edge_attr) == 0:
            edge_attr = torch.tensor(
                [], dtype=torch.float, device=x.device
            ).reshape(num_edges, -1)
        if len(global_features) == 0:
            global_features = torch.tensor(
                [[]], dtype=torch.float, device=x.device
            ).reshape(num_events, -1)

        assert x.dim() == global_features.dim() == edge_attr.dim() == 2
        assert event.dim() == 1
        assert x.shape[1] == self.n_features

        assert global_features.shape[0] == num_events
        assert global_features.shape[1] == self.n_global

        assert edge_attr.shape[0] == num_edges
        if self.msg_nn_include_edge_attr:
            assert edge_attr.shape[1] != 0

        if self.add_self_loops:
            if self.msg_nn_include_edge_attr:
                edge_index, edge_attr = add_self_loops(
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    fill_value=0.0,
                    num_nodes=num_nodes,
                )
            else:
                edge_index, _ = add_self_loops(
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                )

        # Generate a global feature vector in shape of x
        glo_ftx_mtx = global_features[event, :]
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
            if self.msg_nn_include_global:
                xtransform = self.msg_nn(torch.hstack([x, glo_ftx_mtx]))
            else:
                xtransform = self.msg_nn(x)

            new_x = self.propagate(
                edge_index=edge_index,
                edge_attr=edge_attr,  # required, pass as empty
                x=xtransform,
                glo_ftx_mtx=glo_ftx_mtx,  # required, pass as empty
                size=(num_nodes, num_nodes),
            )

        # self loop
        return new_x

    def message(
        self,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        glo_ftx_mtx_j: torch.Tensor,
    ) -> torch.Tensor:
        # This only needs to be done on message level if we consider edge attributes
        if self.msg_nn_bool and self.msg_nn_include_edge_attr:
            msg_parts = [x_j]
            if self.msg_nn_include_global:
                msg_parts.append(glo_ftx_mtx_j)
            if self.msg_nn_include_edge_attr:
                msg_parts.append(edge_attr)
            # Transform node feature matrix with the global features
            xtransform = self.msg_nn(torch.hstack(msg_parts))
        else:
            xtransform = x_j
        return xtransform

    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
        glo_ftx_mtx: torch.Tensor,
    ) -> torch.Tensor:
        if self.upd_nn_bool:
            if self.upd_nn_include_global:
                upd = self.update_nn(torch.hstack([x, glo_ftx_mtx, aggr_out]))
            else:
                upd = self.update_nn(torch.hstack([x, aggr_out]))
        else:
            upd = aggr_out
        return upd
