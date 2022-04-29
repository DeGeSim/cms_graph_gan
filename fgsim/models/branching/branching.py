from math import prod

import torch
import torch.nn as nn

from fgsim.models.ffn import FFN

from .graph_tree import TreeGenType
from .tree import Tree


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
        tree: Tree,
        n_global: int,
        level: int,
        residual: bool = True,
    ):
        super().__init__()
        assert 0 <= level < len(tree.features)
        self.tree = tree
        self.n_branches = self.tree.branches[level]
        self.batch_size = self.tree.batch_size
        self.n_features = self.tree.features[level]
        self.n_global = n_global
        self.level = level
        self.residual = residual

        self.proj_nn = FFN(
            self.n_features + n_global, self.n_features * self.n_branches
        )

    # Split each of the leafs in the the graph.tree into n_branches and connect them
    def forward(self, graph: TreeGenType) -> TreeGenType:
        batch_size = self.batch_size
        n_branches = self.n_branches
        n_features = self.n_features
        parents = self.tree.tree_lists[self.level]
        n_parents = len(parents)

        parents_ftxs = graph.x[graph.idxs_by_level[self.level]]
        device = parents_ftxs.device

        # Compute the new feature vectors:
        parents_idxs = torch.cat([parent.idxs for parent in parents])

        # for the parents indeces generate a matrix where
        # each row is the global vector of the respective event
        if graph.global_features.numel() == 0:
            graph.global_features = torch.empty(
                batch_size, self.n_global, dtype=torch.float, device=device
            )
        parent_global = graph.global_features[parents_idxs % batch_size, :]
        # With the idxs of the parent index the event vector

        # The proj_nn projects the (n_parents * n_event) x n_features to a
        # (n_parents * n_event) x (n_features*n_branches) matrix
        proj_ftx = self.proj_nn(torch.hstack([parents_ftxs, parent_global]))

        # If residual, add the features of the parent to the
        if self.residual:
            proj_ftx = proj_ftx + parents_ftxs.repeat_interleave(
                dim=-1, repeats=n_branches
            )

        assert list(proj_ftx.shape) == [
            n_parents * batch_size,
            n_branches * n_features,
        ]

        children_ftxs = reshape_features(
            proj_ftx,
            n_parents=n_parents,
            batch_size=batch_size,
            n_branches=n_branches,
            n_features=n_features,
        )

        points = prod([br for br in self.tree.branches[: self.level]])
        children = (
            (
                torch.repeat_interleave(
                    torch.arange(batch_size), n_branches * points
                ).reshape(-1, n_branches)
            )
            * n_branches
            * points
        )
        level_idx = torch.arange(
            len(children_ftxs), dtype=torch.long, device=device
        ) + len(graph.x)

        new_graph = TreeGenType(
            x=torch.vstack([graph.x, children_ftxs]),
            idxs_by_level=graph.idxs_by_level + [level_idx],
            children=graph.children + [children],
            edge_index=torch.hstack(self.tree.edge_index_p_level[: self.level + 2]),
            edge_attr=torch.vstack(self.tree.edge_attrs_p_level[: self.level + 2]),
            global_features=graph.global_features,
            batch=torch.arange(batch_size, dtype=torch.long, device=device).repeat(
                (len(graph.x) + len(children_ftxs)) // batch_size
            ),
        )
        return new_graph


# new_graph = Data(
#     x:
#       [currentpoints*batch_size,n_features]
#       full feature vector
#     idxs_by_level:
#       [n_levels]:
#           idxs to that x[idxs_by_level[ilevel]]==levels[ilevel]
#           is the features for the current level
#     children:
#       [n_levels-1]:
#           get the children of the nodes
#           in ilevel via x[idxs_by_level[ilevel+1]][children[ilevel]]
#     edge_index=torch.hstack(self.tree.edge_index_p_level[: self.level + 2]),
#     edge_attr=torch.vstack(self.tree.edge_attrs_p_level[: self.level + 2]),
#     global_features=global_features,
# )


# @torch.jit.script
def reshape_features(
    mtx: torch.Tensor,
    n_parents: int,
    batch_size: int,
    n_branches: int,
    n_features: int,
):
    return (
        mtx.reshape(n_parents, batch_size, n_features * n_branches)
        .transpose(1, 2)
        .reshape(n_parents * n_branches, n_features, batch_size)
        .transpose(1, 2)
        .reshape(n_parents * n_branches * batch_size, n_features)
    )
