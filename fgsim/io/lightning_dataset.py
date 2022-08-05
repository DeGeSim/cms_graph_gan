from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from fgsim.io.queued_dataset import QueuedDataset

from .torch_dataset import PreloadedFromQDS, TrainingFromQDS


class PLDataFromQDS(pl.LightningDataModule):
    def __init__(self, qds: QueuedDataset):
        super().__init__()
        self.qds = qds

    def train_dataloader(self):
        return DataLoader(
            TrainingFromQDS(self.qds),
            collate_fn=None,
            batch_size=None,
            batch_sampler=None,
            n_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            PreloadedFromQDS(self.qds.validation_batches),
            collate_fn=collate_wrapper,
        )

    def test_dataloader(self):
        return DataLoader(
            PreloadedFromQDS(self.qds.testing_batch), collate_fn=collate_wrapper
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        self.qds.qfseq.stop()
        return super().teardown(stage)


def collate_wrapper(batch):
    return batch
