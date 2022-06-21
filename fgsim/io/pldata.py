from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, IterableDataset

# import torch
from fgsim.io.queued_dataset import QueuedDataLoader


class PLDataFromQDL(pl.LightningDataModule):
    def __init__(self, qdl: QueuedDataLoader):
        super().__init__()
        self.qdl = qdl

    def train_dataloader(self):
        return DataLoader(
            TrainingFromQDL(self.qdl),
            collate_fn=None,
            batch_size=None,
            batch_sampler=None,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            PreloadedFromQDL(self.qdl.validation_batches),
            collate_fn=collate_wrapper,
        )

    def test_dataloader(self):
        return DataLoader(
            PreloadedFromQDL(self.qdl.testing_batches), collate_fn=collate_wrapper
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        self.qdl.qfseq.stop()
        return super().teardown(stage)


def collate_wrapper(batch):
    return batch


class TrainingFromQDL(IterableDataset):
    def __init__(self, qdl):
        super().__init__()
        self.qdl = qdl

    def __iter__(self):
        return iter(self.qdl)


class PreloadedFromQDL(IterableDataset):
    def __init__(self, batches):
        super().__init__()
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)
