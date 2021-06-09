import multiprocessing
import time
from pathlib import Path

import h5py as h5
import numpy as np
import torch_geometric

from ..config import conf, device
from ..geo.batch_stack import stack_batch_edge_indexes
from ..geo.transform import transform
from ..utils.logger import logger
from ..utils.thread_or_process import pname
from . import queueflow as qf

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
    file_path, chunk = inp
    logger.debug(f"{pname()}: loading {file_path} chunk {chunk}")
    with h5.File(file_path) as h5_file:
        x = h5_file[conf.loader.xname][chunk[0] : chunk[1]]
        y = h5_file[conf.loader.yname][chunk[0] : chunk[1]]
    logger.debug(f"{pname()}: done loading {file_path} chunk {chunk}")
    return (x, y)


# two processes reading from the filesystem
read_chunk_step = qf.Process_Step(read_chunk, 3, name="read_chunk")
# run until 10 chunks are in the queue

# Step 1.5
# In the input is now [(x,y), ... (x [300 * 51 * 51 * 25], y [300,1] ), (x,y)]
# For these elements to be processed by each of the workers in the following pool
# they need to be (x [51 * 51 * 25], y [1] ):
def zip_chunks(chunks):
    return zip(*chunks)


zip_chunks_step = qf.Process_Step(zip_chunks, 1, name="zip")

# Step 2
# Spawn a Pool with 10 processes for the tranformation from numpy
# Array to Graph
transform_chunk_step = qf.Pool_Step(transform, nworkers=20, name="transform")


## Step 2.5
repack = qf.Repack_Step(conf.loader.batch_size)

## Step 3
def geo_batch(list_of_graphs):
    batch = torch_geometric.data.Batch().from_data_list(list_of_graphs)
    batch = stack_batch_edge_indexes(batch)
    batch = batch
    return batch


geo_batch_step = qf.Process_Step(geo_batch, 2, name="geo_batch")


def to_gpu(batch):
    return batch.to(device)


to_gpu_step = qf.Process_Step(to_gpu, 1, name="to_gpu")

## Collect the steps
process_seq = qf.Sequence(
    read_chunk_step,
    multiprocessing.Queue(5),
    zip_chunks_step,
    multiprocessing.Queue(1),
    transform_chunk_step,
    multiprocessing.Queue(2),
    repack,
    multiprocessing.Queue(1),
    geo_batch_step,
    multiprocessing.Queue(conf.loader.prefetch_batches),
    to_gpu_step,
    multiprocessing.Queue(1),
)


def printflowstatus():
    oldflowstatus = ""
    while True:
        newflowstatus = str(process_seq.flowstatus())
        if newflowstatus != oldflowstatus:
            logger.debug("\n" + newflowstatus)
            oldflowstatus = newflowstatus
        time.sleep(5)


import threading

status_printer_thread = threading.Thread(target=printflowstatus)
status_printer_thread.start()


def get_loader():
    return process_seq(chunk_coords)
