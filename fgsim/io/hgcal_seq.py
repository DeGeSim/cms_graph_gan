"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""
import os
from pathlib import Path
from typing import List, Tuple

import awkward as ak
import queueflow as qf
import torch_geometric
import uproot
import yaml
from torch.multiprocessing import Queue
from torch_geometric.data import Data as GraphType

from fgsim.config import conf
from fgsim.geo.detid_to_graph import event_to_graph

# Load files
ds_path = Path(conf.path.dataset)
assert ds_path.is_dir()
files = sorted(ds_path.glob(conf.path.dataset_glob))
if len(files) < 1:
    raise RuntimeError("No hdf5 datasets found")

ChunkType = List[Tuple[Path, int, int]]

# load lengths
if not os.path.isfile(conf.path.ds_lenghts):
    len_dict = {}
    for fn in files:
        with uproot.open(fn) as rfile:
            len_dict[str(fn)] = rfile[conf.loader.rootprefix].num_entries
    with open(conf.path.ds_lenghts, "w") as f:
        yaml.dump(len_dict, f, Dumper=yaml.SafeDumper)
else:
    with open(conf.path.ds_lenghts, "r") as f:
        len_dict = yaml.load(f, Loader=yaml.SafeLoader)


# reading from the filesystem
def read_chunk(chunks: ChunkType) -> ak.highlevel.Array:
    chunks_list = []
    for chunk in chunks:
        file_path, start, end = chunk
        with uproot.open(file_path) as rfile:
            roottree = rfile[conf.loader.rootprefix]
            chunks_list.append(
                roottree.arrays(
                    conf.loader.keylist,
                    entry_start=start,
                    entry_stop=end,
                    library="ak",
                )
            )

    # split up the events and pass them as a dict
    output = ak.concatenate(chunks_list)

    # remove the double gen energy
    return output


def geo_batch(list_of_graphs: List[GraphType]) -> GraphType:
    batch = torch_geometric.data.Batch().from_data_list(list_of_graphs)
    return batch


ToSparseTranformer = torch_geometric.transforms.ToSparseTensor(
    remove_edge_index=False, fill_cache=True
)


def add_sparse_adj_mtx(batch: GraphType) -> GraphType:
    batch = ToSparseTranformer(batch)
    return batch


def magic_do_nothing(batch: GraphType) -> GraphType:
    return batch


# Collect the steps
def process_seq():
    return (
        qf.ProcessStep(read_chunk, 2, name="read_chunk"),
        qf.PoolStep(
            event_to_graph,
            nworkers=conf.loader.num_workers_transform,
            name="transform",
        ),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(geo_batch, 1, name="geo_batch"),
        qf.ProcessStep(add_sparse_adj_mtx, 1, name="add_sparse_adj_mtx"),
        # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    )
