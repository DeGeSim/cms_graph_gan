import torch
import torch.nn as nn
from torch_geometric.data import Data

from fgsim.config import device

from .tree import Node


class NodeSpliter(nn.Module):
    def __init__(self, n_features: int, n_branches: int, proj_nn: nn.Module):
        super().__init__()
        self.n_features = n_features
        self.n_branches = n_branches
        self.proj_nn = proj_nn

    def forward(self, graph: Data, global_aggr: torch.Tensor) -> Data:
        graph = graph.clone()
        graph.tree.append([])
        last_added_index = len(graph.x) - 1
        new_features = []
        new_edges = []
        new_edge_attrs = []
        # split the nodes in the last layer
        for parent in graph.tree[-2]:
            # Use a NN to do the splitting for each node
            # output is features*splitting
            ftx_children = self.proj_nn(
                torch.hstack([graph.x[parent.idx], global_aggr])
            ).reshape(self.n_branches, self.n_features)
            new_features.append(ftx_children)
            # Calculate the index of the childern
            children_idxs = [last_added_index + i for i in range(self.n_branches)]
            # Connect the parent with the childern
            split_edges = torch.tensor(
                [[parent.idx] * self.n_branches, children_idxs],
                dtype=torch.long,
                device=device,
            )
            new_edges.append(split_edges)
            # write the degree in the edge atts
            new_edge_attrs.append(
                torch.tensor(
                    [[1] * self.n_branches], dtype=torch.long, device=device
                )
            )

            for degree, ancester in enumerate(parent.get_ancestors()):
                ancester_edges = torch.tensor(
                    [[ancester.idx] * self.n_branches, children_idxs],
                    dtype=torch.long,
                    device=device,
                )
                new_edges.append(ancester_edges)
                new_edge_attrs.append(
                    torch.tensor(
                        [[degree + 2] * self.n_branches],
                        dtype=torch.long,
                        device=device,
                    )
                )

            for ichild in range(self.n_branches):
                child = Node(children_idxs[ichild])
                parent.add_child(child)
                graph.tree[-1].append(child)

        graph.x = torch.vstack([graph.x] + new_features)
        graph.edge_index = torch.hstack([graph.edge_index] + new_edges)

        return graph
