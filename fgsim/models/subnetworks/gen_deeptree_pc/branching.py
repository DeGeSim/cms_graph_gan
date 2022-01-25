from math import prod
from typing import List

import torch
import torch.nn as nn
from torch_geometric.data import Data

from .tree import Node


class BranchingLayer(nn.Module):
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
        device = graph.x.device
        # Clone everything to avoid changing the input object
        tree = graph.tree
        x = graph.x.clone()
        edge_index = graph.edge_index.clone()
        edge_attr = graph.edge_attr.clone()
        event = graph.event.clone()
        del graph

        # Add a new tree layer
        tree.append([])
        new_features: List[torch.Tensor] = []
        new_edges: List[torch.Tensor] = []
        new_edge_attrs: List[torch.Tensor] = []
        new_event_idxs: List[torch.Tensor] = []
        # split the nodes in the last layer
        for parent in tree[-2]:
            # Use a NN to do the splitting for each node
            # output is features*splitting
            # #### 1. Compute the new connections ###

            # Calculate the index of the childern
            last_added_index = (
                len(x) + len(new_features) * self.n_branches * self.n_events - 1
            )
            children_idxs = torch.arange(
                last_added_index + 1,
                last_added_index + 1 + self.n_branches * self.n_events,
                dtype=torch.long,
                device=device,
            )
            # ### 3. Make the connections to parent in the node ###
            # Add the child to the tree to keep a reference
            for child_idxs in children_idxs.reshape(self.n_branches, -1):
                child = Node(child_idxs)
                parent.add_child(child)
                tree[-1].append(child)

            # ### 4. Add the connections to the ancestors ###
            for degree, ancestor in enumerate(
                [parent] + parent.get_ancestors(), start=1
            ):
                source_idxs = ancestor.idxs.repeat(self.n_branches)
                ancestor_edges = torch.vstack(
                    [source_idxs, children_idxs],
                )
                new_edges.append(ancestor_edges)
                new_edge_attrs.append(
                    torch.tensor(
                        degree,
                        dtype=torch.long,
                        device=device,
                    ).repeat(self.n_branches * self.n_events)
                )

            # ### 5. New feature vectors ###
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
            # The proj_nn projects the n_event x n_features to a
            # n_event x (n_features*n_branches) matrix
            # For the feature vector of the first branch, we need to slice
            # the first n_features from this output and so on for the
            # other branches
            branch_ftx_list = [
                proj_ftx[
                    ...,
                    self.n_features * (ibranch) : self.n_features * (1 + ibranch),
                ]
                for ibranch in range(self.n_branches)
            ]

            new_features.append(torch.cat(branch_ftx_list))

            # ### 5.b Update the event vector. ###
            # The event vector records which event a feature index belongs
            # to, analogous to torch_geometric.data.Data().batch
            new_event_idxs.append(
                torch.arange(self.n_events, dtype=torch.long, device=device).repeat(
                    self.n_branches
                )
            )

        new_graph = Data(
            x=torch.vstack([x] + new_features),
            edge_index=torch.hstack([edge_index] + new_edges),
            edge_attr=torch.hstack([edge_attr] + new_edge_attrs),
            event=torch.hstack([event] + new_event_idxs),
            tree=tree,
        )

        return new_graph
