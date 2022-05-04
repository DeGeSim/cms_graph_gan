from pathlib import Path
from typing import List, Tuple

import numpy as np
import queueflow as qf
import torch
from torch.multiprocessing import Queue, Value
from torch_geometric.data import Batch, Data

from fgsim.config import conf
from fgsim.io.batch_tools import compute_hlvs
from fgsim.loaders.normal.tree_builder import (
    add_batch_to_branching,
    reverse_construct_tree,
)

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


branches = list(conf.tree.branches)


def transform(_: None) -> Data:
    mu = [1, 1]
    covar = [[1.0, 0.5], [0.5, 1.0]]
    x1 = np.random.multivariate_normal(mu, covar, conf.loader.max_points)
    pointcloud = torch.tensor(x1).float()
    graph = Data(x=pointcloud)
    if postprocess_switch.value:
        graph.hlvs = compute_hlvs(graph)
    branchings_list = cluster_graph(graph, branches)
    graph = reverse_construct_tree(
        graph, branches=branches, branchings_list=branchings_list
    )
    return graph


def aggregate_to_batch(list_of_events: List[Data]) -> Batch:
    batch = Batch.from_data_list(list_of_events)
    batch = add_batch_to_branching(batch, branches, conf.loader.batch_size)
    return batch


def magic_do_nothing(batch: Batch) -> Batch:
    return batch


def cluster_graph(graph: Data, branches: List[int]) -> List[torch.Tensor]:
    cur_nodes = graph.x.shape[0]
    branchings_list = []
    for ilevel, n_branches in reversed(list(enumerate(branches))):
        branching_test_example = torch.tensor(
            np.arange(cur_nodes // n_branches).repeat(n_branches),
            dtype=torch.long,
        )
        branchings_list.append(branching_test_example)
        cur_nodes = cur_nodes // n_branches
    branchings_list.reverse()
    return branchings_list
