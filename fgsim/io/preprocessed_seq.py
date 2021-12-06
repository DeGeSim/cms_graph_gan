"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""
import importlib
from pathlib import Path
from typing import List

import queueflow as qf
import torch
from torch.multiprocessing import Queue

from fgsim.config import conf

# Import the specified processing sequence
sel_seq = importlib.import_module(f"fgsim.io.{conf.loader.qf_seq_name}")

Batch = sel_seq.Batch
DataSetType = List[Batch]

# Load files
dataset_path = Path(conf.path.training)
dataset_path.mkdir(parents=True, exist_ok=True)

# reading from the filesystem
def read_file(file: Path) -> DataSetType:
    batch_list: DataSetType = torch.load(file)
    return batch_list


# Collect the steps
def preprocessed_seq():
    return (
        qf.ProcessStep(read_file, 1, name="read_chunk"),
        qf.pack.UnpackStep(),
        Queue(conf.loader.prefetch_batches),
    )
