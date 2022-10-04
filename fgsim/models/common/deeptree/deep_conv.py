from typing import Optional

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

from fgsim.models.common import FFN


class DeepConv(MessagePassing):
    """
    1. The global features are concatenated with the node
       features and passed to the message generation layer.
    2. The messages are aggregated and passed to the update layer.
    3. The update layer returns the updated node features."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_cond: int,
        n_global: int,
        add_self_loops: bool,
        nns: str,
        msg_nn_include_edge_attr: bool,
        msg_nn_include_global: bool,
        msg_nn_final_linear: bool,
        upd_nn_include_global: bool,
        upd_nn_final_linear: bool,
        residual: bool,
    ):
        super().__init__(aggr="add", flow="source_to_target")
        self.in_features = in_features
        self.out_features = out_features
        self.n_global = n_global
        self.n_cond = n_cond
        self.add_self_loops = add_self_loops
        assert nns in ["msg", "upd", "both"]
        self.msg_nn_bool = nns in ["msg", "both"]
        self.upd_nn_bool = nns in ["upd", "both"]
        self.msg_nn_include_edge_attr = msg_nn_include_edge_attr
        self.residual = residual
        self.msg_nn_final_linear = msg_nn_final_linear
        self.upd_nn_final_linear = upd_nn_final_linear

        if n_global == 0:
            msg_nn_include_global = False
            upd_nn_include_global = False
        self.msg_nn_include_global = msg_nn_include_global
        self.upd_nn_include_global = upd_nn_include_global

        # MSG NN
        self.msg_nn: torch.nn.Module = torch.nn.Identity()
        if self.msg_nn_bool:
            self.msg_nn = FFN(
                in_features
                + (n_global if msg_nn_include_global else 0)
                + (1 if msg_nn_include_edge_attr else 0),
                in_features if self.upd_nn_bool else out_features,
                final_linear=self.msg_nn_final_linear,
            )
        else:
            # assert not (msg_nn_include_edge_attr or msg_nn_include_global)
            pass

        # UPD NN
        self.update_nn: torch.nn.Module = torch.nn.Identity()
        if self.upd_nn_bool:
            self.update_nn = FFN(
                2 * in_features
                + +n_cond
                + (n_global if upd_nn_include_global else 0),
                out_features,
                final_linear=self.upd_nn_final_linear,
            )
        else:
            # assert not upd_nn_include_global
            pass

    def forward(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        global_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_nodes = x.shape[0]

        num_edges = edge_index.shape[1]
        batch_size = int(batch[-1]) + 1
        device = x.device

        if self.n_cond > 0:
            assert cond is not None
            assert cond.shape == (batch_size, self.n_cond)
        else:
            cond = torch.empty(
                batch_size, self.n_cond, dtype=torch.float, device=device
            )

        if self.n_global > 0:
            assert global_features is not None
            assert global_features.shape == (batch_size, self.n_global)

        if edge_attr is None:
            edge_attr = torch.empty(num_edges, 1, dtype=torch.float, device=device)

        assert x.dim() == global_features.dim() == edge_attr.dim() == 2
        assert batch.dim() == 1
        assert x.shape[1] == self.in_features

        assert global_features.shape[0] == batch_size
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
        glo_ftx_mtx = global_features[batch, :]
        glo_cond = cond[batch, :]
        # If the egde_attrs are included, we transforming the message
        if self.msg_nn_include_edge_attr:
            new_x = self.propagate(
                edge_index=edge_index,
                edge_attr=edge_attr,
                x=x,
                cond=glo_cond,
                glo_ftx_mtx=glo_ftx_mtx,
                size=(num_nodes, num_nodes),
            )
            if self.residual:
                new_x[..., : self.in_features] += x[..., : self.out_features]
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
                cond=glo_cond,
                x=xtransform,
                glo_ftx_mtx=glo_ftx_mtx,  # required, pass as empty
                size=(num_nodes, num_nodes),
            )
            if self.residual:
                if self.out_features == self.in_features:
                    new_x = new_x + x
                elif self.out_features < self.in_features:
                    # Downscale: propagate only part of tftx to new_x
                    new_x = new_x + x[..., : self.out_features]
                else:
                    # Upscale: propagate tftx to a part of newx
                    new_x[..., : self.in_features] = (
                        new_x[..., : self.in_features] + x
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
        cond: torch.Tensor,
        glo_ftx_mtx: torch.Tensor,
    ) -> torch.Tensor:
        if self.upd_nn_bool:
            if self.upd_nn_include_global:
                upd = self.update_nn(torch.hstack([x, cond, glo_ftx_mtx, aggr_out]))
            else:
                upd = self.update_nn(torch.hstack([x, cond, aggr_out]))
        else:
            upd = aggr_out
        return upd

    def __repr__(self):
        return f"AncestorConv(msg_nn={self.msg_nn}\n, update_nn={self.update_nn}\n)"
