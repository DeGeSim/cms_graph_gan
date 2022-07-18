from typing import List, Union

import queueflow as qf
import torch
from torch.multiprocessing import Queue, Value
from torch_geometric.data import Batch, Data

from fgsim.config import conf

from .objcol import contruct_graph_from_row, read_chunks, scaler

# Sharded switch for the postprocessing
shared_postprocess_switch = Value("i", 0)
shared_batch_size = Value("i", int(conf.loader.batch_size))


# Collect the steps
def process_seq() -> List[Union[qf.StepBase, Queue]]:
    return [
        qf.ProcessStep(read_chunks, 4, name="read_chunk"),
        qf.PoolStep(
            transform,
            nworkers=conf.loader.num_workers_transform,
            name="transform",
        ),
        qf.RepackStep(shared_batch_size),
        qf.ProcessStep(aggregate_to_batch, 1, name="batch"),
        # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    ]


# Methods used in the Sequence
# reading from the filesystem


def transform(pc) -> Data:
    graph = contruct_graph_from_row(pc)
    graph.x[graph.mask] = torch.from_numpy(
        scaler.transform(graph.x[graph.mask].numpy())
    ).float()
    return graph


def aggregate_to_batch(list_of_events) -> Batch:
    batch = Batch.from_data_list(list_of_events)
    return batch


def magic_do_nothing(batch: Batch) -> Batch:
    return batch
