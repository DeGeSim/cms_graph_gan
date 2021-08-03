import torch
import torch_geometric

from ..config import device


def move_batch_to_device(batch, device):
    def move(element):
        if torch.is_tensor(element):
            return element.to(device)
        elif isinstance(element, list):
            return [move(ee) for ee in element]
        elif isinstance(element, dict):
            return {k: move(ee) for k, ee in element.items()}
        elif element is None:
            return None
        elif isinstance(element, (int, str, float)):
            return element
        else:
            raise ValueError

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
    return batch_new


def clone_or_copy(e):
    if torch.is_tensor(e):
        return e.clone()
    elif isinstance(e, list):
        return [clone_or_copy(ee) for ee in e]
    elif isinstance(e, dict):
        return {k: clone_or_copy(ee) for k, ee in e.items()}
    elif isinstance(e, (int, str, float)):
        return e
    elif e is None:
        return None
    else:
        raise ValueError


def clone_batch(batch):
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
