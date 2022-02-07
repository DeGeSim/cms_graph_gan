import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class AncestorConv(MessagePassing):
    """
    1. The global features are concatenated with the node
       features and passed to the message generation layer.
    2. The messages are aggregated and passed to the update layer.
    3. The update layer returns the updated node features."""

    def __init__(
        self, msg_gen: nn.Module, update_nn: nn.Module, add_self_loops: bool = True
    ):
        super().__init__(aggr="add", flow="source_to_target")
        self.msg_gen = msg_gen
        self.update_nn = update_nn
        self.add_self_loops = add_self_loops
        # self.kappa = torch.nn.Parameter(torch.Tensor([0.95]))

    def forward(
        self,
        *,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        event: torch.Tensor,
        edge_attr: torch.Tensor,
        global_features: torch.Tensor,
    ):
        num_nodes = x.size(0)
        if self.add_self_loops:
            edge_index, edge_attr = add_self_loops(
                edge_index=edge_index,
                edge_attr=edge_attr,
                fill_value=0,
                num_nodes=num_nodes,
            )

        # Generate a global feature vector in shape of x
        glo_ftx_mtx = global_features[event, :]
        new_x = self.propagate(
            edge_index=edge_index,
            edge_attr=edge_attr,
            x=x,
            glo_ftx_mtx=glo_ftx_mtx,
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
        # Transform node feature matrix with the global features
        xtransform = self.msg_gen(
            torch.hstack([x_j, glo_ftx_mtx_j, edge_attr.reshape(x_j.size(0), -1)])
        )
        return xtransform

    def update(
        self,
        aggr_out: torch.Tensor,
        x: torch.Tensor,
        glo_ftx_mtx: torch.Tensor,
    ) -> torch.Tensor:
        return self.update_nn(torch.hstack([x, glo_ftx_mtx, aggr_out]))
