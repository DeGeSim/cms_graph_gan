from typing import List, Optional

import torch
from torch_geometric.data import Data

from fgsim.config import conf, device


class GraphTree(Data):
    def __init__(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        children: Optional[List[torch.Tensor]] = None,
        idxs_by_level: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        if children is None:
            children = []
        if idxs_by_level is None:
            idxs_by_level = [
                torch.arange(
                    conf.loader.batch_size, dtype=torch.long, device=device
                )
            ]

        super().__init__(
            x, children=children, idxs_by_level=idxs_by_level, batch=batch, **kwargs
        )
        self.x_by_level = Levels(self)
        self.batch_by_level = LevelsBatch(self)


class Levels:
    def __init__(self, graphtree: GraphTree) -> None:
        self.graphtree = graphtree

    def __getitem__(self, ilevel):
        return self.graphtree.x[self.graphtree.idxs_by_level[ilevel]]


class LevelsBatch:
    def __init__(self, graphtree: GraphTree) -> None:
        self.graphtree = graphtree

    def __getitem__(self, ilevel):
        return self.graphtree.batch[self.graphtree.idxs_by_level[ilevel]]


class GraphTreeGen(GraphTree):
    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = torch.empty(
            2, 0, dtype=torch.long, device=device
        ),
        edge_attr: torch.Tensor = torch.empty(
            0, 1, dtype=torch.float, device=device
        ),
        batch: torch.Tensor = torch.arange(
            conf.loader.batch_size, dtype=torch.long, device=device
        ),
        global_features: torch.Tensor = torch.empty(
            0, dtype=torch.float, device=device
        ),
        children: Optional[List[torch.Tensor]] = None,
        idxs_by_level: Optional[List[torch.Tensor]] = None,
    ):
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            children=children,
            idxs_by_level=idxs_by_level,
            batch=batch,
            global_features=global_features,
        )
