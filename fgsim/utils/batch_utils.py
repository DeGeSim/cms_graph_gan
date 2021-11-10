from typing import Dict, List

import numpy as np
import torch
import torch_geometric
import torch_sparse

from fgsim.config import device
from fgsim.utils.typecheck import istype


def move_batch_to_device(batch, device):
    """This function moves batches (eg. from torch_geometric) to a specified device
    and also takes into account manually assinged properties."""

    def move(element):
        if torch.is_tensor(element):
            return element.to(device)
        if isinstance(element, torch_sparse.SparseTensor):
            return element.to(device)
        elif isinstance(element, (list, set, tuple)):
            return type(element)((move(ee) for ee in element))
        elif isinstance(element, dict):
            return {k: move(ee) for k, ee in element.items()}
        elif element is None:
            return None
        elif isinstance(element, (int, str, float)):
            return element
        elif type(element).__module__ == np.__name__:
            return element
        else:
            raise ValueError

    if isinstance(batch, torch_geometric.data.Data):
        batch_new = torch_geometric.data.Batch().from_dict(
            {k: move(v) for k, v in batch.to_dict().items()}
        )
        for attr in [
            "__slices__",
            "__cat_dims__",
            "__cumsum__",
            "__num_nodes_list__",
            "__num_graphs__",
        ]:
            if hasattr(batch_new, attr):
                setattr(batch_new, attr, move(getattr(batch, attr)))
    elif istype(batch, torch.Tensor):
        batch_new = batch.to(device)
    elif istype(batch, List[torch.Tensor]):
        batch_new = [move(v) for v in batch]
    elif istype(batch, Dict[str, torch.Tensor]):
        batch_new = {k: move(v) for k, v in batch.items()}
    else:
        raise RuntimeError(
            "Cannot move this object to the torch device, invalid type."
        )
    return batch_new


def clone_or_copy(element):
    if torch.is_tensor(element):
        return element.clone()
    if isinstance(element, torch_sparse.SparseTensor):
        return element.clone()
    elif isinstance(element, (list, set, tuple)):
        return type(element)((clone_or_copy(ee) for ee in element))
    elif isinstance(element, dict):
        return {k: clone_or_copy(ee) for k, ee in element.items()}
    elif isinstance(element, (int, str, float)):
        return element
    elif type(element).__module__ == np.__name__:
        return element
    elif element is None:
        return None
    else:
        raise ValueError


def clone_batch(batch):
    """This function clones batches (eg. from torch_geometric) and
    also takes into account manually assinged properties. This is needed
    when using torch_geometric with torch.multiprocessing"""
    batch_cloned = torch_geometric.data.Batch().from_dict(
        {k: clone_or_copy(v) for k, v in batch.to_dict().items()}
    )

    for attr in [
        "__slices__",
        "__cat_dims__",
        "__cumsum__",
        "__num_nodes_list__",
    ]:
        if hasattr(batch, attr):
            setattr(batch_cloned, attr, clone_or_copy(getattr(batch, attr)))
    return batch_cloned


def check_batch_device(batch):
    for k, v in batch.to_dict().items():
        if type(v) == list:
            for e in v:
                if e.device != device:
                    print(f"Key {k} on wrong device {e.device}")
                    break
        else:
            if v.device != device:
                print(f"Key {k} on wrong device {v.device}")


def batch_to_numpy_dict(batch):
    def tonumpy(element):
        if torch.is_tensor(element):
            return element.numpy()
        elif isinstance(element, list):
            return [tonumpy(ee) for ee in element]
        elif isinstance(element, dict):
            return {k: tonumpy(ee) for k, ee in element.items()}
        elif element is None:
            return None
        elif isinstance(element, (int, str, float)):
            return element
        else:
            raise ValueError

    batch_new = torch_geometric.data.Batch().from_dict(
        {k: tonumpy(v) for k, v in batch.to_dict().items()}
    )
    return batch_new
