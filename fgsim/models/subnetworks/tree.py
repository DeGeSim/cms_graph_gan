from __future__ import annotations

from typing import List, Optional

import torch


class Node:
    def __init__(self, ftx: torch.Tensor):
        self.ftx = ftx
        self.children: List[Node] = []
        self.parent: Optional[Node] = None

    def add_child(self, child: Node):
        assert child.parent is None
        self.children.append(child)
        child.parent = self

    def get_parents(self) -> List[Node]:
        if self.parent is None:
            return []
        else:
            return [self.parent] + self.parent.get_parents()

    def get_root(self) -> Node:
        cur = self
        while cur.parent is not None:
            cur = cur.parent
        return cur
