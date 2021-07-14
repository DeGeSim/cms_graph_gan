import os
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import torch_geometric
from torch.multiprocessing import Queue

from ..config import conf, device
from ..geo.batch_stack import split_layer_subgraphs
from ..geo.transform import transform
from ..utils.logger import logger
from . import qf


# reading from the filesystem
def read_chunk(inp):
    file_path, chunk = inp
    with h5.File(file_path) as h5_file:
        x_vector = h5_file[conf.loader.xname][chunk[0] : chunk[1]]
        y_vector = h5_file[conf.loader.yname][chunk[0] : chunk[1]]
    return (x_vector, y_vector)


def zip_chunks(chunks):
    return zip(*chunks)


def geo_batch(list_of_graphs):
    batch = torch_geometric.data.Batch().from_data_list(list_of_graphs)
    return batch


# Collect the steps
def process_seq():
    return (
        qf.ProcessStep(read_chunk, 1, name="read_chunk"),
        Queue(5),
        # In the input is now [(x,y), ... (x [300 * 51 * 51 * 25], y [300,1] ), (x,y)]
        # For these elements to be processed by each of the workers in the following
        # transformthey need to be (x [51 * 51 * 25], y [1] ):
        qf.ProcessStep(zip_chunks, 1, name="zip"),
        qf.PoolStep(
            transform, nworkers=conf.loader.num_workers_transform, name="transform"
        ),
        Queue(2),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(geo_batch, 1, name="geo_batch"),
        qf.ProcessStep(
            split_layer_subgraphs,
            conf.loader.num_workers_stack,
            name="split_layer_subgraphs",
        ),
        Queue(conf.loader.prefetch_batches),
    )


class QueuedDataLoader:
    def __init__(self):
        ds_path = Path(conf.path.dataset)
        assert ds_path.is_dir()
        self.files = sorted(ds_path.glob("**/*.h5"))
        if len(self.files) < 1:
            raise RuntimeError("No hdf5 datasets found")

        chunksize = conf.loader.chunksize
        nentries = 10000

        chunk_splits = [
            (i * chunksize, (i + 1) * chunksize)
            for i in range(nentries // chunksize)
        ] + [(nentries // chunksize * chunksize, nentries)]

        self.chunk_coords = [
            (fn, chunk_split) for chunk_split in chunk_splits for fn in self.files
        ]

        np.random.shuffle(self.chunk_coords)

        n_validation_batches = (
            conf.loader.validation_set_size // conf.loader.batch_size
        )
        if conf.loader.validation_set_size % conf.loader.batch_size != 0:
            n_validation_batches += 1

        n_test_batches = conf.loader.test_set_size // conf.loader.batch_size
        if conf.loader.test_set_size % conf.loader.batch_size != 0:
            n_test_batches += 1

        logger.info(
            f"Using the first {n_validation_batches} batches for "
            + f"validation and the next {n_test_batches} batches for testing."
        )

        self.validation_chunks = self.chunk_coords[:n_validation_batches]
        self.testing_chunks = self.chunk_coords[
            n_validation_batches : n_validation_batches + n_test_batches
        ]
        self.training_chunks = self.chunk_coords[
            n_validation_batches + n_test_batches :
        ]

        if not os.path.isfile(conf.path.validation):
            logger.warn("Processing validation batches")

            valqfseq = qf.Sequence(*process_seq())
            valqfseq.queue_iterable(self.validation_chunks)
            self.validation_batches = [
                batch.to("cpu").clone().contiguous() for batch in valqfseq
            ]
            torch.save(self.validation_batches, conf.path.validation)
            del valqfseq

        if not os.path.isfile(conf.path.test):
            logger.warn("Processing testing batches")
            testqfseq = qf.Sequence(*process_seq())
            testqfseq.queue_iterable(self.testing_chunks)
            self.testing_batches = [
                batch.to("cpu").clone().contiguous() for batch in testqfseq
            ]
            torch.save(self.testing_batches, conf.path.test)
            del testqfseq

            logger.warn("Validation and training batches pickled.")
        else:
            self.validation_batches = torch.load(
                conf.path.validation, map_location=device
            )

        self.qfseq = qf.Sequence(*process_seq())

    def load_test_batches(self):
        self.testing_batches = torch.load(conf.path.test, map_location=device)

    def queue_epoch(self, n_skip_events=0):
        n_skip_chunks = n_skip_events // conf.loader.chunksize
        n_skip_chunks = n_skip_chunks % (len(self.files) * conf.loader.chunksize)

        logger.info(f" skipping {n_skip_chunks} batches for training.")

        self.epoch_chunks = self.training_chunks[n_skip_chunks:]

        self.qfseq.queue_iterable(self.epoch_chunks)

    def __iter__(self):
        return iter(self.qfseq)
