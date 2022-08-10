from typing import List, Optional, Union

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.nn import knn_graph

# from fgsim.config import conf, device
from fgsim.io.batch_tools import batch_from_pcs_list


# It's a wrapper around a `Data` object that makes it easier
# to access the data at different levels of the tree
class GraphTreeWrapper:
    def __init__(
        self,
        data: Union[Data, Batch],
    ):
        assert data.tftx is not None
        assert data.tbatch is not None
        device = data.tftx.device
        batch_size = int(data.tbatch[-1]) + 1
        if not hasattr(data, "children"):
            data.children = []
        if not hasattr(data, "idxs_by_level"):
            data.idxs_by_level = [
                torch.arange(batch_size, dtype=torch.long, device=device)
            ]
        self.data: Data = data
        self.tftx_by_level = TFTX_BY_LEVEL(self.data)
        self.batch_by_level = BATCH_BY_LEVEL(self.data)

    def __repr__(self) -> str:
        return f"GraphTreeWrapper({self.data})"

    # assign all the stuff the the data member
    def __setattr__(self, __name, __value):
        # avoid overwriting the existing members
        if __name in ["data", "tftx_by_level", "batch_by_level", "pc"]:
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
            return self.tftx_by_level[-1]
        elif __name == "batch_by_level":
            return self.batch_by_level
        elif __name == "data":
            return self.data
        elif __name == "tftx_by_level":
            return self.tftx_by_level
        else:
            return getattr(self.data, __name)

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)


# It's a wrapper around the `data.tftx` array that allows you to index it by level
class TFTX_BY_LEVEL:
    def __init__(self, data: Data) -> None:
        self.data = data

    def __getitem__(self, slice):
        return self.data.tftx[self.data.idxs_by_level[slice]]


# It's a wrapper around a Data object that allows you to index into the Data object by level
class BATCH_BY_LEVEL:
    def __init__(self, data: Data) -> None:
        self.data = data

    def __getitem__(self, slice):
        return self.data.tbatch[self.data.idxs_by_level[slice]]


# > This class is used to store the type of tree generation
class TreeGenType(Data):
    def __init__(
        self,
        tftx: torch.Tensor,
        batch_size: int,
        # edge_index: torch.Tensor = torch.empty(2, 0, dtype=torch.long),
        # edge_attr: torch.Tensor = torch.empty(0, 1, dtype=torch.float),
        global_features: torch.Tensor = torch.empty(0, dtype=torch.float),
        children: Optional[List[torch.Tensor]] = None,
        idxs_by_level: Optional[List[torch.Tensor]] = None,
        cur_level: int = 0,
    ):
        device = tftx.device

        tbatch = torch.arange(batch_size, dtype=torch.long, device=device).repeat(
            (len(tftx)) // batch_size
        )
        super().__init__(
            tftx=tftx,
            # edge_index=edge_index,
            # edge_attr=edge_attr,
            children=children,
            idxs_by_level=idxs_by_level,
            tbatch=tbatch,
            global_features=global_features,
            cur_level=cur_level,
        )
        self = self.to(device)


def graph_tree_to_batch(
    batch: Union[Data, TreeGenType, GraphTreeWrapper], n_nn: int = 0
) -> Batch:
    """
    It takes a batch of graphs and returns a batch of graphs

    Args:
      batch (Union[Batch, TreeGenType, GraphTreeWrapper]):
      a batch of graphs, either a Batch object,
      a TreeGenType object, or a GraphTreeWrapper object.
      n_nn (int): number of nearest neighbors to use for the graph.
      If 0, no graph is used. Defaults to 0

    Returns:
      A batch object with the edge_index attribute set
      to the k-nearest neighbors of the nodes in the graph.
    """
    if isinstance(batch, GraphTreeWrapper):
        res = batch_from_pcs_list(
            batch.tftx_by_level[-1],
            batch.batch_by_level[-1],
        )
    elif hasattr(batch, "idxs_by_level"):
        graph_tree = GraphTreeWrapper(batch)
        res = batch_from_pcs_list(
            graph_tree.tftx_by_level[-1],
            graph_tree.batch_by_level[-1],
        )
    if n_nn > 1:
        res.edge_index = knn_graph(x=res.x, k=n_nn, batch=res.batch)
    return res
