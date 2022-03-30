import torch
import torch.nn as nn
from torch_geometric.data import Data

from fgsim.models.dnn_gen import dnn_gen

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
        n_features: int,
        n_global: int,
        level: int,
        residual: bool = True,
    ):
        super().__init__()
        self.tree = tree
        self.n_branches = tree.branches[level]
        self.batch_size = self.tree.batch_size
        self.n_features = n_features
        self.n_global = n_global
        self.level = level
        self.residual = residual
        assert 1 <= level < len(tree.branches)
        self.proj_nn = dnn_gen(
            self.n_features + n_global, self.n_features * self.n_branches
        )

    # Split each of the leafs in the the graph.tree into n_branches and connect them
    def forward(self, graph: Data) -> Data:
        device = graph.x.device
        # Clone everything to avoid changing the input object

        x = graph.x.clone()
        global_features = graph.global_features.clone()
        del graph

        batch_size = self.batch_size
        n_branches = self.n_branches
        n_features = self.n_features
        parents = self.tree.tree_lists[self.level - 1]
        n_parents = len(parents)

        edge_index_p_level = self.tree.edge_index_p_level
        edge_attrs_p_level = self.tree.edge_attrs_p_level

        # Compute the new feature vectors:
        parents_idxs = torch.cat([parent.idxs for parent in parents])
        # for the parents indeces generate a matrix where
        # each row is the global vector of the respective event
        parent_global = global_features[parents_idxs % batch_size, :]
        # With the idxs of the parent index the event vector
        parents_ftxs = x[parents_idxs, ...]

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

        new_graph = Data(
            x=torch.cat([x, children_ftxs]),
            edge_index=torch.hstack(edge_index_p_level[: self.level + 1]),
            edge_attr=torch.vstack(edge_attrs_p_level[: self.level + 1]),
            global_features=global_features,
        )
        new_graph.event = torch.arange(
            batch_size, dtype=torch.long, device=device
        ).repeat(len(new_graph.x) // batch_size)
        return new_graph


@torch.jit.script
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
