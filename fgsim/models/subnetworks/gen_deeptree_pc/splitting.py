from math import prod

import torch
import torch.nn as nn
from torch_geometric.data import Data

from fgsim.config import device

from .tree import Node


class NodeSpliter(nn.Module):
    """Splits the last set of Nodes of a given graph.
    Order for x : node>event>branch
    Example with 3 events,2 branches for a single split:
    FeatureIndex[I]/Event[E]/Branch[B]
    F|E|B
    -|-|-
    0|0|0
    1|1|0
    2|2|0
    3|0|0
    4|0|1
    5|1|0
    6|1|1
    7|2|0
    8|2|1
    """

    def __init__(
        self, n_events: int, n_features: int, n_branches: int, proj_nn: nn.Module
    ):
        super().__init__()
        self.n_events = n_events
        self.n_features = n_features
        self.n_branches = n_branches
        self.proj_nn = proj_nn

    # Split each of the leafs in the the graph.tree into n_branches and connect them
    def forward(self, graph: Data, global_features: torch.Tensor) -> Data:
        # Clone everything to avoid changing the input object
        tree = [[Node(ee.idxs.clone()) for ee in e] for e in graph.tree]
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()
        edge_attr = graph.edge_attr.clone()
        event = graph.event.clone()
        del graph

        tree.append([])
        last_added_index = len(x) - 1
        new_features = []
        new_edges = []
        new_edge_attrs = []
        new_event_idxs = []
        # split the nodes in the last layer
        for parent in tree[-2]:
            # Use a NN to do the splitting for each node
            # output is features*splitting

            # for the parents indeces generate a matrix where
            # each row is the global vector of the respective event:
            # With the idxs of the parent index the event vector
            # event[parent.idxs] (events of the parent indices)
            parent_global = global_features[event[parent.idxs], :]
            parent_ftx_mtx = x[parent.idxs, :]

            proj_ftx = self.proj_nn(torch.hstack([parent_ftx_mtx, parent_global]))
            # check the projection for the corrert number of entries
            assert prod(parent_ftx_mtx.shape) * self.n_branches == prod(
                proj_ftx.shape
            )

            # the feature vectors for the children are in a row
            # so we do some reshaping
            ftx_children = proj_ftx.reshape(-1, self.n_features)
            # assert that all the children for one event are
            # after on another. Order for x : TreeNode>Event>Branch
            assert torch.all(
                proj_ftx[0]
                == torch.hstack([ftx_children[i] for i in range(self.n_branches)])
            )
            assert torch.all(
                proj_ftx[-1]
                == torch.hstack(
                    [ftx_children[i] for i in range(-self.n_branches, 0)]
                )
            )

            new_features.append(ftx_children)

            # Mark to which event the generated points belong
            # Should be like [0,0,0,1,1,1,2,2,2,3,3,3,...]
            new_event_idxs.append(
                torch.arange(self.n_events, dtype=torch.long, device=device)
                .repeat(self.n_branches, 1)
                .T.reshape(-1)
            )

            # Should be like
            # [parent0,parent0,parent0,
            # ,parent1,parent1,parent1,
            # parent2,parent2,parent2...]
            parent_idxs = parent.idxs.repeat(self.n_branches, 1).T.reshape(-1)

            # Calculate the index of the childern
            children_idxs = torch.arange(
                last_added_index + 1,
                last_added_index + 1 + self.n_branches * self.n_events,
                dtype=torch.long,
                device=device,
            )
            # Connect the parent with the childern
            split_edges = torch.vstack([parent_idxs, children_idxs])
            new_edges.append(split_edges)

            # write the degree in the edge atts
            new_edge_attrs.append(
                torch.tensor(
                    [[1] * self.n_branches * self.n_events],
                    dtype=torch.long,
                    device=device,
                )
            )

            for degree, ancester in enumerate(parent.get_ancestors()):
                ancester_edges = torch.tensor(
                    [[ancester.idxs] * self.n_branches, children_idxs],
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

            for child_idxs in children_idxs.reshape(self.n_branches, -1):
                child = Node(child_idxs)
                parent.add_child(child)
                tree[-1].append(child)

        new_graph = Data(
            x=torch.vstack([x] + new_features),
            edge_index=torch.hstack([edge_index] + new_edges),
            edge_attr=torch.hstack([edge_attr] + new_edge_attrs),
            event=torch.hstack([event] + new_event_idxs),
            tree=tree,
        )

        return new_graph
