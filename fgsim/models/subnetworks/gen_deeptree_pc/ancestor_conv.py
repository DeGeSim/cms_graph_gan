import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


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

    def forward(self, graph, global_features):
        x = graph.x
        edge_index = graph.edge_index

        # Construct the per-event-feature vector for each
        # row in the feature matrix
        glo_ftx_mtx = global_features[graph.event, :]
        # Transform node feature matrix with the global features
        xtransform = self.msg_gen(torch.hstack([x, glo_ftx_mtx]))

        # start propagating messages.
        x = self.propagate(
            edge_index, x=xtransform, xorginal=x, glo_ftx_mtx=glo_ftx_mtx
        )
        return x

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

    def update(
        self,
        aggr_out: torch.Tensor,
        xorginal: torch.Tensor,
        glo_ftx_mtx: torch.Tensor,
    ) -> torch.Tensor:
        return self.update_nn(torch.hstack([aggr_out, xorginal, glo_ftx_mtx]))
