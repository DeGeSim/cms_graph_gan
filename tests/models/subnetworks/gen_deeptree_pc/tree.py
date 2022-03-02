import pytest
import torch

from fgsim.models.branching.node import Node


# The tree is initialized with a root node, which has two children.
# The root node has two children, each of which have two children.
class ExampleTree:
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
def tree() -> ExampleTree:
    return ExampleTree()


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
