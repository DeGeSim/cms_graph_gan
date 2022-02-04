import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj


# 1. The global features are concatenated with the node
#    features and passed to the message generation layer.
# 2. The messages are aggregated and passed to the update layer.
# 3. The update layer returns the updated node features.
class AncestorConvLayer(MessagePassing):
    def __init__(self, msg_gen: nn.Module, update_nn: nn.Module):
        super().__init__(aggr="add", flow="source_to_target")
        self.msg_gen = msg_gen
        self.update_nn = update_nn
        self.kappa = torch.nn.Parameter(torch.Tensor([0.95]))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Adj,
        event: torch.Tensor,
        global_features: torch.Tensor,
    ):
        glo_ftx_mtx = global_features[event, :]
        # x_pair = (xtransform, xtransform)
        # start propagating messages.
        # x = self.propagate(edge_index, x=x_pair, glo_ftx_mtx=glo_ftx_mtx)
        new_x = self.propagate(edge_index=edge_index, x=x, glo_ftx_mtx=glo_ftx_mtx)

        # self loop
        return new_x

    def message(self, x_j: torch.Tensor, glo_ftx_mtx: torch.Tensor) -> torch.Tensor:
        # Construct the per-event-feature vector for each
        # row in the feature matrix

        # Transform node feature matrix with the global features
        xtransform = self.msg_gen(torch.hstack([x_j, glo_ftx_mtx]))
        return xtransform

    def update(
        self,
        aggr_out: torch.Tensor,
        x_i: torch.Tensor,
        # xorginal: torch.Tensor,
        glo_ftx_mtx: torch.Tensor,
    ) -> torch.Tensor:
        # return self.update_nn(torch.hstack([aggr_out, xorginal, glo_ftx_mtx]))
        return self.update_nn(torch.hstack([aggr_out]))
