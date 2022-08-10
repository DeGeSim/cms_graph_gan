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
        branches = self.branches
        batch_size = self.batch_size
        n_levels = len(self.features)
        device = self.device
        # initialize the root
        # shape : 2 x num_edges
        self.edge_index_p_level: List[torch.Tensor] = [
            torch.empty(2, 0, dtype=torch.long, device=device)
        ]
        # shape : num_edges x 1
        self.edge_attrs_p_level: List[torch.Tensor] = [
            torch.empty(0, 1, dtype=torch.long, device=device)
        ]
        self.tree_lists: List[List[Node]] = [
            [Node(torch.arange(self.batch_size, dtype=torch.long, device=device))]
        ]
        next_x_index = self.batch_size

        # Start with 1 because the root is initialized
        for level in range(1, n_levels):
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
                #     branches[level-1] * batch_size * len(self.tree_lists[level])
                # )
                # points_used_cur = branches[level-1] * batch_size * iparent
                # next_x_index = (
                #     last_idx_in_prev_level
                #     + points_in_cur_level
                #     + points_used_cur
                #     + 1
                # )

                children_idxs = torch.arange(
                    next_x_index,
                    next_x_index + branches[level - 1] * batch_size,
                    dtype=torch.long,
                    device=device,
                )
                next_x_index = next_x_index + branches[level - 1] * batch_size
                # ### 3. Make the connections to parent in the node ###
                # Add the child to the self.tree to keep a reference
                for child_idxs in children_idxs.reshape(branches[level - 1], -1):
                    child = Node(child_idxs)
                    parent.add_child(child)
                    self.tree_lists[-1].append(child)

                # ### 4. Add the connections to the ancestors ###
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
            self.edge_index_p_level.append(torch.hstack(new_edges))
            self.edge_attrs_p_level.append(torch.vstack(new_edge_attrs))

    def __repr__(self) -> str:
        outstr = ""
        attr_list = [
            "branches",
            "features",
            "tree_lists",
            "edge_index_p_level",
            "edge_attrs_p_level",
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
