import torch
from torch_geometric.data import Batch

from fgsim.config import conf

from ..base_dataset import BaseDS
from .graph_transform import event_to_graph
from .readin import file_manager, read_chunks
from .scaler import scaler


class Dataset(BaseDS):
    def __init__(self):
        super().__init__(file_manager)

    def _chunk_to_batch(self, chunks):
        batch = events_to_batch(chunks)
        return batch


def events_to_batch(chks: tuple[torch.Tensor, torch.Tensor]):
    graph_list = [
        scale_event(event_to_graph([ey, ex])) for ey, ex in read_chunks(chks)
    ]
    batch = Batch.from_data_list(graph_list)
    return batch


def scale_event(graph):
    graph.n_pointsv = (
        graph.y[..., conf.loader.y_features.index("num_particles")]
        .int()
        .reshape(-1)
    )
    graph.x = scaler.transform(graph.x, "x")
    graph.y = scaler.transform(graph.y, "y")
    return graph
