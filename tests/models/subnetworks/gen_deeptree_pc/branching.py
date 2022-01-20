from typing import List

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from fgsim.config import device
from fgsim.models.subnetworks.gen_deeptree_pc.branching import BranchingLayer
from fgsim.models.subnetworks.gen_deeptree_pc.tree import Node

n_features = 4
n_branches = 2
n_global = 6
n_events = 3


@pytest.fixture
def graph():
    g = Data(
        x=torch.randn(n_events, n_features, dtype=torch.float, device=device),
        edge_index=torch.tensor([[], []], dtype=torch.long, device=device),
        edge_attr=torch.tensor([], dtype=torch.long, device=device),
        event=torch.arange(n_events, dtype=torch.long, device=device),
        tree=[[Node(torch.arange(n_events, dtype=torch.long, device=device))]],
    )
    return g


def test_NodeSpliter(graph: Data):
    branching_layer = BranchingLayer(
        n_events=n_events,
        n_features=n_features,
        n_branches=n_branches,
        proj_nn=nn.Sequential(
            nn.Linear(n_features + n_global, n_features * n_branches),
            nn.ReLU(),
        ),
    ).to(device)
    global_features = torch.randn(
        n_events, n_global, dtype=torch.float, device=device
    )
    for isplit in range(4):
        # ### Splitting
        graph = branching_layer(graph, global_features)

        n_parents = len(graph.tree[-2])
        assert len(graph.tree[-1]) == n_parents * n_branches

        # x shape testing
        assert graph.x.shape[1] == n_features
        assert len(graph.tree) == isplit + 2
        assert graph.x.shape[0] == n_events * sum(
            [n_branches ** i for i in range(len(graph.tree))]
        )
        # edge_index shape testing
        assert graph.edge_index.shape[0] == 2
        # Number of connections
        # Sum n_branches^ilayer*ilayer for ilayer in 0..nlayers
        assert graph.edge_index.shape[1] == n_events * sum(
            [n_branches ** i * i for i in range(len(graph.tree))]
        )

        conlist = graph.edge_index.T.cpu().numpy().tolist()
        connections = {tuple(x) for x in conlist}
        # No double connections
        assert len(connections) == len(conlist)

        # Static Check
        if isplit == 0:
            # F|E|B
            # -|-|-
            # 0|0|0
            # 1|1|0
            # 2|2|0
            # 3|0|0
            # 4|0|1
            # 5|1|0
            # 6|1|1
            # 7|2|0
            # 8|2|1
            # Check the event mapping
            assert np.all(graph.event.cpu().numpy() == [0, 1, 2, 0, 1, 2, 0, 1, 2])
            # check the connections
            expected_connections = {(0, 3), (1, 4), (0, 6), (1, 7), (2, 5), (2, 8)}
            assert connections.issuperset(expected_connections)
        if isplit == 1:
            expected_connections = {
                (3, 9),
                (3, 12),
                (4, 10),
                (4, 13),
                (5, 11),
                (5, 14),
                (6, 15),
                (6, 18),
                (7, 16),
                (7, 19),
                (8, 17),
                (8, 20),
            }
            assert connections.issuperset(expected_connections)

    conlist = graph.edge_index.T.cpu().numpy().tolist()
    connections = {tuple(x) for x in conlist}

    def recurr_check_connection(node: Node, ancestors_idxs: List[np.ndarray]):
        new_ancestors_idxs = [node.idxs.cpu().numpy()] + ancestors_idxs
        for idxs in new_ancestors_idxs:
            for child in node.children:
                child_idxs = child.idxs.cpu().numpy()
                for source, target in zip(
                    idxs,
                    child_idxs,
                ):
                    assert (source, target) in connections
                recurr_check_connection(child, new_ancestors_idxs)

    recurr_check_connection(graph.tree[0][0], [])
