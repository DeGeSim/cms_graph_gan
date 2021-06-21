from pathlib import Path

import h5py as h5
import numpy as np
import torch_geometric
from torch.multiprocessing import Queue

from ..config import conf, device
from ..geo.batch_stack import split_layer_subgraphs
from ..geo.transform import transform
from ..utils.logger import logger
from ..utils.thread_or_process import pname
from . import queueflow as qf

# %%
# Search for all h5 files


# Step 1
def read_chunk(inp):
    file_path, chunk = inp
    logger.debug(f"{pname()}: loading {file_path} chunk {chunk}")
    with h5.File(file_path) as h5_file:
        x = h5_file[conf.loader.xname][chunk[0] : chunk[1]]
        y = h5_file[conf.loader.yname][chunk[0] : chunk[1]]
    logger.debug(f"{pname()}: done loading {file_path} chunk {chunk}")
    return (x, y)


# two processes reading from the filesystem
read_chunk_step = qf.Process_Step(read_chunk, 1, name="read_chunk")
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
transform_chunk_step = qf.Pool_Step(
    transform, nworkers=conf.loader.transform_workers, name="transform"
)


# Step 2.5
repack = qf.Repack_Step(conf.loader.batch_size)

# Step 3
def geo_batch(list_of_graphs):
    batch = torch_geometric.data.Batch().from_data_list(list_of_graphs)
    return batch


geo_batch_step = qf.Process_Step(geo_batch, 1, name="geo_batch")


split_layer_subgraphs_step = qf.Process_Step(
    split_layer_subgraphs, 1, name="split_layer_subgraphs"
)


def to_gpu(batch):
    return batch.to(device)


to_gpu_step = qf.Process_Step(to_gpu, 1, name="to_gpu")

# Collect the steps
process_seq = qf.Sequence(
    read_chunk_step,
    Queue(5),
    zip_chunks_step,
    Queue(1),
    transform_chunk_step,
    Queue(2),
    repack,
    Queue(1),
    geo_batch_step,
    Queue(1),
    split_layer_subgraphs_step,
    Queue(conf.loader.prefetch_batches),
    to_gpu_step,
    Queue(1),
)


class queued_data_loader:
    def __init__(self):
        p = Path(conf.datasetpath)
        assert p.is_dir()
        self.files = sorted(p.glob("**/*.h5"))
        if len(self.files) < 1:
            raise RuntimeError("No hdf5 datasets found")

        chunksize = conf.loader.chunksize
        nentries = 10000

        chunk_splits = [
            (i * chunksize, (i + 1) * chunksize) for i in range(nentries // chunksize)
        ] + [(nentries // chunksize * chunksize, nentries)]

        self.chunk_coords = [
            (fn, chunk_split) for chunk_split in chunk_splits for fn in self.files
        ]

        np.random.shuffle(self.chunk_coords)

        n_validation_batches = conf.loader.validation_set_size // conf.loader.batch_size
        if conf.loader.validation_set_size % conf.loader.batch_size != 0:
            n_validation_batches += 1

        logger.info(f"Using the first {n_validation_batches} batches for validation.")

        self.validation_chunks = self.chunk_coords[:n_validation_batches]
        self.training_chunks = self.chunk_coords[n_validation_batches:]

    def start_epoch(self, n_skip_events=0):
        n_skip_chunks = n_skip_events // conf.loader.chunksize
        n_skip_chunks = n_skip_chunks % (len(self.files) * conf.loader.chunksize)

        logger.info(f" skipping {n_skip_chunks} batches for training.")

        self.epoch_chunks = self.training_chunks[n_skip_chunks:]
        self.epoch_gen = process_seq(self.epoch_chunks)

    def get_epoch_generator(self):
        assert hasattr(self, "epoch_gen"), "Call start_epoch first"
        return self.epoch_gen

    def get_validation_batches(self):
        if not hasattr(self, "validation_batches"):
            qfseq = process_seq(self.validation_chunks)
            # get the validation batches out
            self.validation_batches = [batch.to("cpu") for batch in qfseq]
        return self.validation_batches
