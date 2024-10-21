from caloutils.utils.batch import add_graph_attr

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


def scale_event(batch):
    n_pointsv = (
        batch.y[..., conf.loader.y_features.index("num_particles")]
        .int()
        .reshape(-1)
    )
    add_graph_attr(batch, "n_pointsv", n_pointsv)

    batch.x = scaler.transform(batch.x, "x")
    batch.y = scaler.transform(batch.y, "y")

    return batch
