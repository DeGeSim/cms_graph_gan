import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from fgsim.config import device
from fgsim.models.subnetworks.gen_deeptree_pc.ancester_conv import AncestorConv
from fgsim.models.subnetworks.gen_deeptree_pc.global_feedback import GlobalDeepAggr
from fgsim.models.subnetworks.gen_deeptree_pc.splitting import NodeSpliter
from fgsim.models.subnetworks.gen_deeptree_pc.tree import Node

n_features = 4
n_branches = 2
n_global = 6
batch_size = 3


@pytest.fixture
def graph():
    g = Data(
        x=torch.randn(batch_size, n_features, dtype=torch.float, device=device),
        edge_index=torch.tensor([[], []], dtype=torch.long, device=device),
        edge_attr=torch.tensor([], dtype=torch.long, device=device),
        event=torch.arange(batch_size, dtype=torch.long, device=device),
        tree=[[Node(torch.arange(batch_size, dtype=torch.long, device=device))]],
    )

    return g


def test_GlobalFeedBackNN(graph):
    global_aggr = GlobalDeepAggr(
        pre_nn=nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(),
        ),
        post_nn=nn.Sequential(
            nn.Linear(n_features, n_global),
            nn.ReLU(),
        ),
    ).to(device)
    splitter = NodeSpliter(
        batch_size=batch_size,
        n_features=n_features,
        n_branches=n_branches,
        proj_nn=nn.Sequential(
            nn.Linear(n_features + n_global, n_features * n_branches),
            nn.ReLU(),
        ),
    ).to(device)
    ancester_conv = AncestorConv(
        msg_gen=nn.Sequential(
            nn.Linear(n_features + n_global, n_features),
            nn.ReLU(),
        ),
        update_nn=nn.Sequential(
            # agreegated features + previous feature vector + global
            nn.Linear(2 * n_features + n_global, n_features),
            nn.ReLU(),
        ),
    ).to(device)
    for isplit in range(4):
        # ### Global
        global_features = global_aggr(graph)
        assert global_features.shape[1] == n_global

        # ### Splitting
        # record previous shapes
        old_x_shape = torch.tensor(graph.x.shape)
        old_edge_index_shape = torch.tensor(graph.edge_index.shape)

        graph = splitter(graph, global_features)
        assert graph.x.shape[1] == n_features

        n_parents = len(graph.tree[-2])
        assert len(graph.tree[-1]) == n_parents * n_branches

        # x shape testing
        # add n_parents * n_branches rows to the feature matrix
        assert torch.all(
            torch.tensor([n_parents * n_branches * batch_size, 0]) + old_x_shape
            == torch.tensor(graph.x.shape)
        )
        # edge_index shape testing
        # add one connection for each brach and each parent
        assert torch.all(
            torch.tensor([0, n_parents * n_branches * batch_size])
            + old_edge_index_shape
            == torch.tensor(graph.edge_index.shape)
        )
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
            assert np.all(graph.event.cpu().numpy() == [0, 1, 2, 0, 0, 1, 1, 2, 2])
            assert np.all(
                graph.edge_index.cpu().numpy()
                == [[0, 0, 1, 1, 2, 2], [3, 4, 5, 6, 7, 8]]
            )
        if isplit == 1:
            assert np.all(
                graph.edge_index[:, 6:].cpu().numpy()
                == [
                    [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
                    [9, 10, 11, 12, 13, 14, 9, 10, 11, 12, 13, 14],
                ]
            )

        # ### Convolution
        graph.x = ancester_conv(graph, global_features)


# The tree is initialized with a root node, which has two children.
# The root node has two children, each of which have two children.
class Tree:
    def __init__(self):
        #   /b -d
        # a-
        #   \c -e
        self.a = Node(torch.tensor(0, dtype=torch.long))
        self.b = Node(torch.tensor(1, dtype=torch.long))
        self.c = Node(torch.tensor(2, dtype=torch.long))
        self.d = Node(torch.tensor(3, dtype=torch.long))
        self.e = Node(torch.tensor(4, dtype=torch.long))

        self.a.add_child(self.b)
        self.a.add_child(self.c)

        self.b.add_child(self.d)
        self.c.add_child(self.e)


@pytest.fixture
def tree() -> Tree:
    return Tree()


class TestTree:
    def test_get_root(self, tree):
        assert tree.d.get_root() == tree.a
        assert tree.e.get_root() == tree.a

    def test_get_ancestors(self, tree):
        assert tree.a.get_ancestors() == []
        assert tree.b.get_ancestors() == [tree.a]
        assert tree.d.get_ancestors() == [tree.b, tree.a]

    def test_recur_descendants(self, tree):
        assert tree.d.recur_descendants() == []
        assert tree.b.recur_descendants() == [tree.d]
        assert set(tree.a.recur_descendants()) == set(
            [tree.b, tree.c, tree.d, tree.e]
        )

    def test_get_node_list(self, tree):
        assert set(tree.a.get_node_list()) == set(
            [
                tree.a,
                tree.b,
                tree.c,
                tree.d,
                tree.e,
            ]
        )
