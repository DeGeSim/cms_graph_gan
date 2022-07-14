from pathlib import Path
from typing import List, Tuple, Union

import awkward as ak
import queueflow as qf
import torch
from torch.multiprocessing import Queue, Value
from torch_geometric.data import Batch
from torch_geometric.data import Data as Data

from fgsim.config import conf
from fgsim.io.batch_tools import compute_hlvs

from .objcol import read_chunks, scaler
from .transform import hitlist_to_pc

# Sharded switch for the postprocessing
postprocess_switch = Value("i", 0)


ChunkType = List[Tuple[Path, int, int]]


# Regular sequence of processing train/test/validation
def process_seq() -> List[Union[qf.StepBase, Queue]]:
    return [
        qf.ProcessStep(read_chunks, 2, name="read_chunk"),
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
    ]


def aggregate_to_batch(list_of_events: List[Data]) -> Batch:
    batch = Batch.from_data_list(list_of_events)
    return batch


def magic_do_nothing(batch: Batch) -> Batch:
    return batch


def transform(hitlist: ak.highlevel.Record) -> Data:
    pointcloud = hitlist_to_pc(hitlist)
    graph = Data(x=pointcloud)
    if postprocess_switch.value:
        graph.hlvs = compute_hlvs(graph)
    graph.x = torch.from_numpy(scaler.transform(graph.x.numpy())).float()
    return graph
