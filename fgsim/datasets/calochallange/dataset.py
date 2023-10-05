from fgsim.config import conf

from ..base_dataset import BaseDS
from .graph_transform import events_to_batch
from .readin import file_manager
from .scaler import scaler


class Dataset(BaseDS):
    def __init__(self):
        super().__init__(file_manager)

    def _chunk_to_batch(self, chunks):
        batch = scale_event(events_to_batch(chunks))
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
