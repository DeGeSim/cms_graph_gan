from pathlib import Path
from typing import List, Tuple

import pandas as pd
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
        qf.ProcessStep(read_chunk, 4, name="read_chunk"),
        qf.PoolStep(
            transform,
            nworkers=conf.loader.num_workers_transform,
            name="transform",
        ),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(aggregate_to_batch, 1, name="batch"),
        # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    )


# Methods used in the Sequence
# reading from the filesystem
def read_chunk(chunks: ChunkType):
    chunks_list = []
    for chunk in chunks:
        file_path, start, end = chunk
        chunks_list.append(
            pd.read_csv(
                file_path,
                skiprows=start,
                nrows=(end - start),
                sep=",",
                # header=0,
                # index_col=0,
            )
        )
    res = pd.concat(chunks_list).values.reshape(-1, 30, 3)
    # res = res[..., :3]
    return torch.tensor(res).float()


def transform(pc):
    graph = Data(x=pc)
    if postprocess_switch.value:
        graph.hlvs = compute_hlvs(graph)
    return graph


def aggregate_to_batch(list_of_events) -> Batch:
    batch = Batch.from_data_list(list_of_events)
    return batch


def magic_do_nothing(batch: Batch) -> Batch:
    return batch
