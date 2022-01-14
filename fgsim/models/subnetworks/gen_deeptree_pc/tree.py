from __future__ import annotations

from typing import List, Optional


class Node:
    def __init__(self, idx: int):
        self.idx: int = idx
        self.children: List[Node] = []
        self.parent: Optional[Node] = None

    def add_child(self, child: Node):
        assert child.parent is None
        self.children.append(child)
        child.parent = self

    def get_ancestors(self) -> List[Node]:
        if self.parent is None:
            return []
        else:
            return [self.parent] + self.parent.get_ancestors()

    def recur_descendants(self):
        return self.children + [
            coc for child in self.children for coc in child.recur_descendants()
        ]

    def get_root(self) -> Node:
        cur = self
        while cur.parent is not None:
            cur = cur.parent
        return cur

    def get_node_list(self) -> List[Node]:
        root = self.get_root()
        node_list = [root] + root.recur_descendants()
        return node_list
