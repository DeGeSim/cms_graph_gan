from math import prod
from typing import List

import torch

from .node import Node


class Tree:
    def __init__(
        self,
        batch_size: int,
        branches: List[int],
        features: List[int],
        device: torch.device,
    ):
        self.batch_size = batch_size
        self.branches = branches
        self.features = features
        self.device = device

        assert len(self.features) - 1 == len(self.branches)
        n_levels = len(self.features)
        # initialize the root
        # shape : 2 x num_edges
        self.ancestor_edge_index_p_level: List[torch.Tensor] = [
            torch.empty(2, 0, dtype=torch.long, device=device)
        ]
        # shape : num_edges x 1
        self.ancestor_edge_attrs_p_level: List[torch.Tensor] = [
            torch.empty(0, 1, dtype=torch.long, device=device)
        ]
        # shape : 2 x num_edges
        self.children_edge_index_p_level: List[torch.Tensor] = [
            torch.empty(2, 0, dtype=torch.long, device=device)
        ]

        self.tree_lists: List[List[Node]] = [
            [Node(torch.arange(self.batch_size, dtype=torch.long, device=device))]
        ]

        self.tbatch_by_level: List[torch.Tensor] = [
            torch.arange(batch_size, dtype=torch.long, device=device).repeat(
                prod([branches[:ilevel]])
            )
            for ilevel in range(1, n_levels)
        ]
        next_x_index = self.batch_size

        # Start with 1 because the root is initialized
        for level in range(1, n_levels):
            # Add a new tree layer
            self.tree_lists.append([])
            new_edges: List[torch.Tensor] = []
            new_children_edges: List[torch.Tensor] = []
            new_edge_attrs: List[torch.Tensor] = []

            # split the nodes in the previous layer
            for iparent, parent in enumerate(self.tree_lists[level - 1]):
                children_idxs = torch.arange(
                    next_x_index,
                    next_x_index + branches[level - 1] * batch_size,
                    device=device,
                ).long()
                next_x_index = next_x_index + branches[level - 1] * batch_size
                # ### Make the connections to parent in the node ###
                # Add the child to the self.tree to keep a reference
                for child_idxs in children_idxs.reshape(branches[level - 1], -1):
                    child = Node(child_idxs)
                    parent.add_child(child)
                    self.tree_lists[-1].append(child)
                    # Dense connetions within the children
                new_children_edges.append(
                    self._children_dense_ei_from_parent(parent)
                )

                # ### Add the connections to the ancestors ###
                for degree, ancestor in enumerate(
                    [parent] + parent.get_ancestors(), start=1
                ):
                    source_idxs = ancestor.idxs.repeat(branches[level - 1])
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
                        .repeat(branches[level - 1] * batch_size)
                        .reshape(-1, 1)
                    )
            self.ancestor_edge_index_p_level.append(torch.hstack(new_edges))
            self.children_edge_index_p_level.append(
                torch.hstack(new_children_edges)
            )
            self.ancestor_edge_attrs_p_level.append(torch.vstack(new_edge_attrs))

    def _children_dense_ei_from_parent(self, parent: Node) -> torch.Tensor:
        children_idxs = torch.vstack([child.idxs for child in parent.children])
        n_children = len(parent.children)
        # This expression will produce dense conncetions with the child nodes of a batch
        # Plus a self loop
        # children_idxs [ [0,1],[2,3] ]
        # -> [[0,0,1,1,2,2,3,3],
        #     [0,1,0,1,2,3,2,3]]
        return torch.vstack(
            [
                children_idxs.repeat_interleave(n_children, 0).reshape(-1),
                children_idxs.repeat(n_children, 1).reshape(-1),
            ]
        )

    def ancestor_ei(self, ilevel):
        return torch.hstack(self.ancestor_edge_index_p_level[: ilevel + 1])

    def ancestor_ea(self, ilevel):
        return torch.vstack(self.ancestor_edge_attrs_p_level[: ilevel + 1])

    def children_ei(self, ilevel):
        return torch.hstack(self.children_edge_index_p_level[: ilevel + 1])

    def __repr__(self) -> str:
        outstr = ""
        attr_list = [
            "branches",
            "features",
            "tree_lists",
            "ancestor_edge_index_p_level",
            "ancestor_edge_attrs_p_level",
            "children_edge_index_p_level",
        ]
        for attr_name in attr_list:
            attr = getattr(self, attr_name)
            if attr_name == "tree_lists":
                outstr += f"  {attr_name}("
                for ilevel, treelevel in enumerate(attr):
                    outstr += f"    level {ilevel}: {printtensor(treelevel)}),\n"
                outstr += f"),\n"
            else:
                outstr += f"  {attr_name}({printtensor(attr)}),\n"
        return f"Tree(\n{outstr})"


def printtensor(obj):
    if isinstance(obj, torch.Tensor):
        return repr(obj.shape)
    elif isinstance(obj, list):
        return repr([printtensor(e) for e in obj])
    elif isinstance(obj, dict):
        return repr({k: printtensor(v) for k, v in obj.items()})
    else:
        return repr(obj)
