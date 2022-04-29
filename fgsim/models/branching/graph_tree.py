from typing import List, Optional

import torch
from torch_geometric.data import Data

from fgsim.config import conf, device


# It's a wrapper around a `Data` object that makes it easier
# to access the data at different levels of the tree
class GraphTreeWrapper:
    def __init__(
        self,
        data: Data,
    ):
        assert hasattr(data, "x")
        assert hasattr(data, "batch")
        if not hasattr(data, "children"):
            data.children = []
        if not hasattr(data, "idxs_by_level"):
            data.idxs_by_level = [
                torch.arange(
                    conf.loader.batch_size, dtype=torch.long, device=device
                )
            ]
        self.data: Data = data
        self.x_by_level = X_BY_LEVEL(self.data)
        self.batch_by_level = BATCH_BY_LEVEL(self.data)

    # assign all the stuff the the data member
    def __setattr__(self, __name, __value):
        # avoid overwriting the existing members
        if __name in ["data", "x_by_level", "batch_by_level", "pc"]:
            #  should be assigned on init
            if __name in self.__dict__ or __name == "pc":
                raise AttributeError(f"Cannot assign {__name} to the Wrapper")
            else:
                self.__dict__[__name] = __value
        else:
            setattr(self.data, __name, __value)

    # wrap getattr to the data member
    def __getattr__(self, __name):
        if __name == "pc":
            return self.x_by_level[-1]
        elif __name == "batch_by_level":
            return self.batch_by_level
        elif __name == "data":
            return self.data
        elif __name == "x_by_level":
            return self.x_by_level
        else:
            return getattr(self.data, __name)

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)


# It's a wrapper around the `data.x` array that allows you to index it by level
class X_BY_LEVEL:
    def __init__(self, data: Data) -> None:
        self.data = data

    def __getitem__(self, slice):
        return self.data.x[self.data.idxs_by_level[slice]]


# It's a wrapper around a Data object that allows you to index into the Data object by level
class BATCH_BY_LEVEL:
    def __init__(self, data: Data) -> None:
        self.data = data

    def __getitem__(self, slice):
        return self.data.batch[self.data.idxs_by_level[slice]]


# > This class is used to store the type of tree generation
class TreeGenType(Data):
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
