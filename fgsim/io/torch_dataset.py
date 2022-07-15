from torch.utils.data import IterableDataset

from fgsim.io.queued_dataset import QueuedDataset


class TrainingFromQDS(IterableDataset):
    def __init__(self, qds: QueuedDataset):
        super().__init__()
        self.qds = qds

    def __iter__(self):
        return iter(self.qds)


class PreloadedFromQDS(IterableDataset):
    def __init__(self, batches):
        super().__init__()
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)
