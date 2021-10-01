"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""
from pathlib import Path
from typing import List

import torch
from torch.multiprocessing import Queue
from torch_geometric.data import Data as GraphType

from fgsim.config import conf
from fgsim.io import qf

# Load files
dataset_path = Path(conf.path.training)
dataset_path.mkdir(parents=True, exist_ok=True)

# reading from the filesystem
def read_file(file: Path) -> List[GraphType]:
    batch_list: List[GraphType] = torch.load(file)
    return batch_list


# Collect the steps
def preprocessed_seq():
    return (
        qf.ProcessStep(read_file, 1, name="read_chunk"),
        qf.pack.UnpackStep(),
        Queue(conf.loader.prefetch_batches),
    )
