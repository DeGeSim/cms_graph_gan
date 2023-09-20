import torch
from torch_geometric.data import Batch, Data


def event_to_graph(chk: tuple[torch.Tensor, torch.Tensor]) -> Data:
    y, x = chk
    res = Data(
        x=x[x[..., 3].bool(), :3].reshape(-1, 3),
        y=y.reshape(1, -1),
        n_pointsv=x[..., 3].sum().long(),
    )
    return res


def events_to_batch(chks: tuple[torch.Tensor, torch.Tensor]):
    batch = Batch.from_data_list([event_to_graph([ey, ex]) for ey, ex in chks])
    return batch
