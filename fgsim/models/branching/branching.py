import torch
import torch.nn as nn
from torch_geometric.data import Data

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
        proj_nn: nn.Module,
    ):
        super().__init__()
        self.proj_nn = proj_nn
        self.tree = tree

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

        tree = self.tree.tree_lists
        n_parents = len(tree[isplit])
        n_events = self.tree.n_events
        n_branches = self.tree.n_branches
        n_features = self.tree.n_features
        edge_index_p_level = self.tree.edge_index_p_level
        edge_attrs_p_level = self.tree.edge_attrs_p_level

        # Compute the new feature vectors:
        parents_idxs = torch.cat([parent.idxs for parent in tree[isplit]])
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
            edge_index=torch.hstack(edge_index_p_level[: isplit + 1]),
            edge_attr=torch.hstack(edge_attrs_p_level[: isplit + 1]),
            isplit=isplit + 1,
        )
        new_graph.event = torch.arange(
            n_events, dtype=torch.long, device=device
        ).repeat(len(new_graph.x) // n_events)
        return new_graph


@torch.jit.script
def reshape_features(
    mtx: torch.Tensor,
    n_parents: int,
    n_events: int,
    n_branches: int,
    n_features: int,
):
    return (
        mtx.reshape(n_parents, n_events, n_features * n_branches)
        .transpose(1, 2)
        .reshape(n_parents * n_branches, n_features, n_events)
        .transpose(1, 2)
        .reshape(n_parents * n_branches * n_events, n_features)
    )
