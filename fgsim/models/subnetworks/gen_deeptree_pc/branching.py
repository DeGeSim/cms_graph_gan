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
        self,
        *,
        proj_nn: nn.Module,
        n_events: int,
        n_features: int,
        n_branches: int,
        n_levels: int,
        device: torch.device,
    ):
        super().__init__()
        self.n_events = n_events
        self.n_features = n_features
        self.n_branches = n_branches
        self.n_levels = n_levels
        self.proj_nn = proj_nn
        self.device = device

        self.edge_index_p_level: List[torch.Tensor] = []
        self.edge_attrs_p_level: List[torch.Tensor] = []
        self.tree: List[List[Node]] = [
            [Node(torch.arange(n_events, dtype=torch.long, device=device))]
        ]

        for level in range(self.n_levels):
            # Add a new tree layer
            self.tree.append([])
            new_edges: List[torch.Tensor] = []
            new_edge_attrs: List[torch.Tensor] = []

            # split the nodes in the last layer
            for iparent, parent in enumerate(self.tree[level]):
                # Use a NN to do the splitting for each node
                # output is features*splitting
                # #### 1. Compute the new connections ###

                # Calculate the index of the childern
                last_idx_in_prev_level = (
                    2 if 2 == len(self.tree) else self.tree[level - 1][-1].idxs[-1]
                )
                points_in_cur_level = (
                    n_branches * n_events * len(self.tree[level - 1])
                )
                points_used_cur = n_branches * n_events * iparent
                next_x_index = (
                    last_idx_in_prev_level
                    + points_in_cur_level
                    + points_used_cur
                    + 1
                )

                children_idxs = torch.arange(
                    next_x_index,
                    next_x_index + n_branches * n_events,
                    dtype=torch.long,
                    device=device,
                )
                # ### 3. Make the connections to parent in the node ###
                # Add the child to the self.tree to keep a reference
                for child_idxs in children_idxs.reshape(n_branches, -1):
                    child = Node(child_idxs)
                    parent.add_child(child)
                    self.tree[-1].append(child)

                # ### 4. Add the connections to the ancestors ###
                for degree, ancestor in enumerate(
                    [parent] + parent.get_ancestors(), start=1
                ):
                    source_idxs = ancestor.idxs.repeat(n_branches)
                    ancestor_edges = torch.vstack(
                        [source_idxs, children_idxs],
                    )
                    new_edges.append(ancestor_edges)
                    new_edge_attrs.append(
                        torch.tensor(
                            degree,
                            dtype=torch.long,
                            device=device,
                        ).repeat(n_branches * n_events)
                    )
            self.edge_index_p_level.append(torch.hstack(new_edges))
            self.edge_attrs_p_level.append(torch.hstack(new_edge_attrs))

    # Split each of the leafs in the the graph.tree into n_branches and connect them
    def forward(self, graph: Data, global_features: torch.Tensor) -> Data:
        device = graph.x.device
        # Clone everything to avoid changing the input object

        x = graph.x.clone()
        if not hasattr(graph, "isplit"):
            isplit = 0
        else:
            isplit = graph.isplit
        del graph

        n_parents = len(self.tree[isplit])
        n_events = self.n_events
        n_branches = self.n_branches
        n_features = self.n_features

        # Compute the new feature vectors:
        parents_idxs = torch.cat([parent.idxs for parent in self.tree[isplit]])
        # for the parents indeces generate a matrix where
        # each row is the global vector of the respective event
        parent_global = global_features[parents_idxs % n_events, :]
        # With the idxs of the parent index the event vector
        parents_ftxs = x[parents_idxs, ...]

        # The proj_nn projects the (n_parents * n_event) x n_features to a
        # (n_parents * n_event) x (n_features*n_branches) matrix
        proj_ftx = self.proj_nn(torch.hstack([parents_ftxs, parent_global]))

        assert list(proj_ftx.shape) == [
            n_parents * n_events,
            n_branches * n_features,
        ]

        children_ftxs = reshape_features(
            proj_ftx,
            n_parents=n_parents,
            n_events=n_events,
            n_branches=n_branches,
            n_features=n_features,
        )
        new_graph = Data(
            x=torch.cat([x, children_ftxs]),
            edge_index=torch.hstack(self.edge_index_p_level[: isplit + 1]),
            edge_attr=torch.hstack(self.edge_attrs_p_level[: isplit + 1]),
            isplit=isplit + 1,
        )
        new_graph.event = torch.arange(
            self.n_events, dtype=torch.long, device=device
        ).repeat(len(new_graph.x) // self.n_events)
        return new_graph


@torch.jit.script
def reshape_features(
    mtx: torch.Tensor,
    n_parents: int,
    n_events: int,
    n_branches: int,
    n_features: int,
):
    magic_list = [colmtx.split(n_features, dim=1) for colmtx in mtx.split(1, dim=0)]

    out_list = []
    for iparent in range(n_parents):
        for ibranch in range(n_branches):
            for ievent in range(n_events):
                row = ievent + iparent * n_events
                out_list.append(magic_list[row][ibranch])
    return torch.cat(out_list)
