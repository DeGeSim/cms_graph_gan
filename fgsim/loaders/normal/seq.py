from pathlib import Path
from typing import List, Tuple

import numpy as np
import queueflow as qf
import torch
from torch.multiprocessing import Queue, Value
from torch_geometric.data import Batch, Data

from fgsim.config import conf
from fgsim.io.batch_tools import compute_hlvs

# Sharded switch for the postprocessing
postprocess_switch = Value("i", 0)

ChunkType = List[Tuple[Path, int, int]]
# Collect the steps
def process_seq():
    return (
        qf.ProcessStep(read_chunk, 2, name="read_chunk"),
        qf.PoolStep(
            transform,
            nworkers=conf.loader.num_workers_transform,
            name="transform",
        ),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(aggregate_to_batch, 1, name="batch"),
        # # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    )


# Methods used in the Sequence
# reading from the filesystem
def read_chunk(chunks: ChunkType) -> List[None]:
    return [None for chunk in chunks for _ in range(chunk[2] - chunk[1])]


def transform(_: None) -> Data:
    mu = [1, 1]
    covar = [[1.0, 0.5], [0.5, 1.0]]
    x1 = np.random.multivariate_normal(mu, covar, conf.loader.max_points)
    pointcloud = torch.tensor(x1).float()
    graph = Data(x=pointcloud)
    if postprocess_switch.value:
        graph.hlvs = compute_hlvs(graph)
    return graph


def aggregate_to_batch(list_of_events: List[Data]) -> Batch:
    batch = Batch.from_data_list(list_of_events)
    return batch


def magic_do_nothing(batch: Batch) -> Batch:
    return batch