from typing import Optional, Union

import torch
from torch_geometric.data import Batch, Data

# from fgsim.config import conf, device
from fgsim.io.batch_tools import batch_from_pcs_list

from .tree import Tree


# It's a wrapper around a `Data` object that makes it easier
# to access the data at different levels of the tree
class GraphTreeWrapper:
    def __init__(self, data: Union[Data, Batch], tree: Tree):
        assert data.tftx is not None
        self.__dict__["__init"] = True
        self.tree = tree
        self.data: Data = data
        self.tftx_by_level = TFTX_BY_LEVEL(self.data, self.tree)
        self.batch_by_level = BATCH_BY_LEVEL(self.data, self.tree)
        self.__dict__["__init"] = False

    def __repr__(self) -> str:
        return f"GraphTreeWrapper({self.data})"

    # assign all the stuff the the data member
    def __setattr__(self, __name, __value):
        # after init write to the data member
        if self.__dict__["__init"]:
            self.__dict__[__name] = __value
        else:
            setattr(self.data, __name, __value)

    # # wrap getattr to the data member
    def __getattr__(self, __name):
        if __name in self.__dict__:
            return self.__dict__[__name]
        else:
            return getattr(self.__dict__["data"], __name)

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def get_batch_skeleton(self) -> Batch:
        """
        It takes a batch of graphs and returns the skeleton batch of graphs

        Returns:
        A batch object with the edge_index

        """
        last_level_batch_idx = self.tree.tbatch_by_level[-1][
            self.tree.idxs_by_level[-1]
        ]
        res: Batch = batch_from_pcs_list(
            self.tftx_by_level[-1],
            last_level_batch_idx,
        )
        self.__presaved_batch: Batch = res.detach().clone()
        self.__presaved_batch.x = None
        self.__presaved_batch_indexing: torch.Tensor = torch.argsort(
            last_level_batch_idx
        )
        return self.__presaved_batch, self.__presaved_batch_indexing
        # if isinstance(batch, GraphTreeWrapper):
        #     res = batch_from_pcs_list(
        #         batch.tftx_by_level[-1],
        #         batch.batch_by_level[-1],
        #     )
        # elif hasattr(batch, "idxs_by_level"):
        #     graph_tree = GraphTreeWrapper(batch)
        #     res = batch_from_pcs_list(
        #         graph_tree.tftx_by_level[-1],
        #         graph_tree.batch_by_level[-1],
        #     )


# It's a wrapper around the `data.tftx` array that allows you to index it by level
class TFTX_BY_LEVEL:
    def __init__(self, data: Data, tree: Tree) -> None:
        self.data = data
        self.tree = tree

    def __getitem__(self, slice):
        return self.data.tftx[self.tree.idxs_by_level[slice]]


# It's a wrapper around a Data object that allows you to index into the Data object by level
class BATCH_BY_LEVEL:
    def __init__(self, data: Data, tree: Tree) -> None:
        self.data = data
        self.tree = tree

    def __getitem__(self, slice):
        return self.tree.tbatch[self.tree.idxs_by_level[slice]]


# > This class is used to store the type of tree generation
class TreeGenType(Data):
    def __init__(
        self,
        tftx: torch.Tensor,
        batch_size: int,
        cond: Optional[torch.Tensor] = None,
        # edge_index: torch.Tensor = torch.empty(2, 0, dtype=torch.long),
        # edge_attr: torch.Tensor = torch.empty(0, 1, dtype=torch.float),
        global_features: torch.Tensor = torch.empty(0, dtype=torch.float),
        # children: Optional[List[torch.Tensor]] = None,
        # idxs_by_level: Optional[List[torch.Tensor]] = None,
        cur_level: int = 0,
    ):
        device = tftx.device

        # tbatch = torch.arange(batch_size, dtype=torch.long, device=device).repeat(
        #     (len(tftx)) // batch_size
        # )
        super().__init__(
            tftx=tftx,
            # edge_index=edge_index,
            # edge_attr=edge_attr,
            cond=cond,
            # children=children,
            # idxs_by_level=idxs_by_level,
            # tbatch=tbatch,
            global_features=global_features,
            cur_level=cur_level,
        )
        self = self.to(device)
