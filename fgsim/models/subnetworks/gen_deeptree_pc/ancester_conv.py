import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class AncesterConv(MessagePassing):
    def __init__(self, msg_gen: nn.Module, update_nn: nn.Module):
        super().__init__(aggr="add", flow="source_to_target")
        self.msg_gen = msg_gen
        self.update_nn = update_nn
        self.kappa = torch.nn.Parameter(torch.Tensor([0.95]))

    def forward(self, graph, global_features):
        x = graph.x
        edge_index = graph.edge_index

        # Transform node feature matrix with the global features
        xt = self.msg_gen(torch.hstack([x, global_features.repeat(x.shape[0], 1)]))

        # start propagating messages.
        x = self.propagate(edge_index, x=x, xt=xt, global_features=global_features)
        return x

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j

    def update(self, aggr_out: torch.Tensor, x, global_features) -> torch.Tensor:
        return self.update_nn(
            torch.hstack([aggr_out, x, global_features.repeat(x.shape[0], 1)])
        )
