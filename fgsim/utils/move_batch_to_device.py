import torch
import torch_geometric


def move_batch_to_device(batch, device):
    def move(element):
        if torch.is_tensor(element):
            return element.to(device)
        elif isinstance(element, list):
            return [move(ee) for ee in element]
        elif element is None:
            return None
        else:
            raise ValueError

    batch_new = torch_geometric.data.Batch().from_dict(
        {k: move(v) for k, v in batch.to_dict().items()}
    )
    return batch_new
