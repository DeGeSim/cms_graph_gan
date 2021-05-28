import multiprocessing
from pathlib import Path

import h5py as h5
import numpy as np
import torch_geometric

from ..config import conf
from ..geo.transform import transform
from ..utils.logger import logger
from ..utils.thread_or_process import pname
from .process_queue import MP_Pipe_Process_Step, MP_Pipe_Sequence

# %%
# Search for all h5 files
p = Path(conf.datasetpath)
assert p.is_dir()
files = sorted(p.glob("**/*.h5"))
if len(files) < 1:
    raise RuntimeError("No hdf5 datasets found")


chunksize = conf.loader.chunksize
nentries = 10000

chunk_splits = [
    (i * chunksize, (i + 1) * chunksize) for i in range(nentries // chunksize)
] + [(nentries // chunksize * chunksize, nentries)]

chunk_coords = [(fn, chunk_split) for chunk_split in chunk_splits for fn in files]

np.random.shuffle(chunk_coords)

## Step 1
def read_chunk(inp):
    file_path, chunk=inp
    logger.debug(f"{pname()}: loading {file_path} chunk {chunk}")
    with h5.File(file_path) as h5_file:
        x = h5_file[conf.loader.xname][chunk[0] : chunk[1]]
        y = h5_file[conf.loader.yname][chunk[0] : chunk[1]]
    logger.debug(f"{pname()}: done loading {file_path} chunk {chunk}")
    return (x, y)


# two processes reading from the filesystem
read_chunk_step = MP_Pipe_Process_Step(read_chunk, 4)
# run until 10 chunks are in the queue
chunks_queue = multiprocessing.Queue(10)


## Step 2
def transform_chunk(chunks):
    with multiprocessing.Pool(5) as p:
        return p.map(transform, zip(*chunks))


# 2x5 processes for the transformation
transform_chunk_step = MP_Pipe_Process_Step(transform_chunk, 2, deamonize = False)

list_of_graphs_queue = multiprocessing.Queue(4)

## Step 3
def geo_batch(list_of_graphs):
    return torch_geometric.data.Batch().from_data_list(list_of_graphs)


geo_batch_step = MP_Pipe_Process_Step(geo_batch, 2)

batch_prefetch_queue = multiprocessing.Queue(10)

## Collect the steps
process_seq = MP_Pipe_Sequence(
    read_chunk_step,
    chunks_queue,
    transform_chunk_step,
    list_of_graphs_queue,
    geo_batch_step,
    batch_prefetch_queue,
)


def get_loader():
    return process_seq(chunk_coords)
