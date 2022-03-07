from dataclasses import dataclass
from typing import List

import torch

from .node import Node


@dataclass
class Tree:
    n_events: int
    n_features: int
    n_branches: int
    n_levels: int
    device: torch.device

    def __post_init__(self):
        n_branches = self.n_branches
        n_events = self.n_events
        device = self.device
        # initialize the root
        # shape : 2 x num_edges
        self.edge_index_p_level: List[torch.Tensor] = [
            torch.tensor([[], []], dtype=torch.long, device=device)
        ]
        # shape : num_edges x 1
        self.edge_attrs_p_level: List[torch.Tensor] = [
            torch.tensor([], dtype=torch.long, device=device).reshape(0, 1)
        ]
        self.tree_lists: List[List[Node]] = [
            [Node(torch.arange(self.n_events, dtype=torch.long, device=device))]
        ]
        next_x_index = self.n_events

        # Start with 1 because the root is initialized
        for level in range(1, self.n_levels):
            # Add a new tree layer
            self.tree_lists.append([])
            new_edges: List[torch.Tensor] = []
            new_edge_attrs: List[torch.Tensor] = []

            # split the nodes in the previous layer
            for iparent, parent in enumerate(self.tree_lists[level - 1]):
                # Use a NN to do the splitting for each node
                # output is features*splitting
                # #### 1. Compute the new connections ###

                # # Calculate the index of the childern
                # last_idx_in_prev_level = int(self.tree_lists[level - 1][-1].idxs[-1])
                # points_in_cur_level = (
                #     n_branches * n_events * len(self.tree_lists[level])
                # )
                # points_used_cur = n_branches * n_events * iparent
                # next_x_index = (
                #     last_idx_in_prev_level
                #     + points_in_cur_level
                #     + points_used_cur
                #     + 1
                # )

                children_idxs = torch.arange(
                    next_x_index,
                    next_x_index + n_branches * n_events,
                    dtype=torch.long,
                    device=device,
                )
                next_x_index = next_x_index + n_branches * n_events
                # ### 3. Make the connections to parent in the node ###
                # Add the child to the self.tree to keep a reference
                for child_idxs in children_idxs.reshape(n_branches, -1):
                    child = Node(child_idxs)
                    parent.add_child(child)
                    self.tree_lists[-1].append(child)

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
                        )
                        .repeat(n_branches * n_events)
                        .reshape(-1, 1)
                    )
            self.edge_index_p_level.append(torch.hstack(new_edges))
            self.edge_attrs_p_level.append(torch.vstack(new_edge_attrs))
