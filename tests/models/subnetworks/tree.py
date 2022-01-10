import pytest
import torch

from fgsim.models.subnetworks.tree import Node


class Tree:
    def __init__(self):
        #   /b -d
        # a-
        #   \c -e
        self.a = Node(torch.tensor(0))
        self.b = Node(torch.tensor(1))
        self.c = Node(torch.tensor(2))
        self.d = Node(torch.tensor(3))
        self.e = Node(torch.tensor(4))

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

    def test_get_parents(self, tree):
        assert tree.a.get_parents() == []
        assert tree.b.get_parents() == [tree.a]
        assert tree.d.get_parents() == [tree.b, tree.a]
