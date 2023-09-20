from ..base_dataset import BaseDS
from .graph_transform import events_to_batch
from .readin import file_manager, read_chunks
from .scaler import scaler


class Dataset(BaseDS):
    def __init__(self):
        super().__init__(file_manager)

    def _chunk_to_batch(self, chunks):
        batch = events_to_batch(read_chunks(chunks))
        batch.x = scaler.transform(batch.x, "x")
        batch.y = scaler.transform(batch.y, "y")
        return batch
