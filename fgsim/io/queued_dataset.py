import os
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import torch_geometric
import yaml
from torch.multiprocessing import Queue

from ..config import conf, device
from ..geo.batch_stack import split_layer_subgraphs
from ..geo.transform import transform
from ..utils.logger import logger
from . import qf


# reading from the filesystem
def read_chunk(chunkL):
    data_dict = {k: [] for k in conf.loader.keylist}
    for chunk in chunkL:
        file_path, start, end = chunk
        with h5.File(file_path) as h5_file:
            for k in conf.loader.keylist:
                data_dict[k].append(h5_file[k][start:end])
    for k in conf.loader.keylist:
        if len(data_dict[k][0].shape) == 1:
            data_dict[k] = np.hstack(data_dict[k])
        else:
            data_dict[k] = np.vstack(data_dict[k])

    # split up the events and pass them as a dict
    output = [
        {k: data_dict[k][ientry] for k in conf.loader.keylist}
        for ientry in range(conf.loader.chunksize)
    ]
    return output


def geo_batch(list_of_graphs):
    batch = torch_geometric.data.Batch().from_data_list(list_of_graphs)
    return batch


def magic_do_nothing(x):
    return x


# Collect the steps
def process_seq():
    return (
        qf.ProcessStep(read_chunk, 2, name="read_chunk"),
        Queue(1),
        # In the input is now [(x,y), ... (x [300 * 51 * 51 * 25], y [300,1] ), (x,y)]
        # For these elements to be processed by each of the workers in the following
        # transformthey need to be (x [51 * 51 * 25], y [1] ):
        qf.PoolStep(
            transform, nworkers=conf.loader.num_workers_transform, name="transform"
        ),
        Queue(1),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(geo_batch, 1, name="geo_batch"),
        qf.ProcessStep(
            split_layer_subgraphs,
            conf.loader.num_workers_stack,
            name="split_layer_subgraphs",
        ),
        # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    )


ds_path = Path(conf.path.dataset)
assert ds_path.is_dir()
files = [str(e) for e in sorted(ds_path.glob("**/*.h5"))]
if len(files) < 1:
    raise RuntimeError("No hdf5 datasets found")


# load lengths
if not os.path.isfile(conf.path.ds_lenghts):
    len_dict = {}
    for fn in files:
        with h5.File(fn) as h5_file:
            len_dict[fn] = len(h5_file[conf.yvar])
    with open(conf.path.ds_lenghts, "w") as f:
        yaml.dump(len_dict, f, Dumper=yaml.SafeDumper)
else:
    with open(conf.path.ds_lenghts, "r") as f:
        len_dict = yaml.load(f, Loader=yaml.SafeLoader)


class QueuedDataLoader:
    def __init__(self):
        chunksize = conf.loader.chunksize
        batch_size = conf.loader.batch_size

        chunk_coords = [[]]
        ifile = 0
        ielement = 0
        current_chunck_elements = 0
        while ifile < len(files):
            elem_left_in_cur_file = len_dict[files[ifile]] - ielement
            elem_to_add = chunksize - current_chunck_elements
            if elem_left_in_cur_file > elem_to_add:
                chunk_coords[-1].append(
                    [files[ifile], ielement, ielement + elem_to_add]
                )
                ielement += elem_to_add
                current_chunck_elements += elem_to_add
            else:
                chunk_coords[-1].append(
                    [files[ifile], ielement, ielement + elem_left_in_cur_file]
                )
                ielement = 0
                current_chunck_elements += elem_left_in_cur_file
                ifile += 1
            if current_chunck_elements == chunksize:
                current_chunck_elements = 0
                chunk_coords.append([])

        # remove the last, uneven chunk
        chunk_coords = list(
            filter(
                lambda chunk: sum([part[2] - part[1] for part in chunk])
                == chunksize,
                chunk_coords,
            )
        )

        np.random.shuffle(chunk_coords)

        # Make sure the chunks can be split evenly into batches:
        assert chunksize % batch_size == 0

        assert conf.loader.validation_set_size % chunksize == 0
        n_validation_batches = conf.loader.validation_set_size // batch_size
        n_validation_chunks = conf.loader.validation_set_size // chunksize

        assert conf.loader.test_set_size % chunksize == 0
        n_test_batches = conf.loader.test_set_size // batch_size
        n_testing_chunks = conf.loader.test_set_size // chunksize

        logger.info(
            f"Using the first {n_validation_batches} batches for "
            + f"validation and the next {n_test_batches} batches for testing."
        )

        self.validation_chunks = chunk_coords[:n_validation_chunks]
        self.testing_chunks = chunk_coords[
            n_validation_chunks : n_validation_chunks + n_testing_chunks
        ]
        self.training_chunks = chunk_coords[
            n_validation_chunks + n_testing_chunks :
        ]

        self.qfseq = qf.Sequence(*process_seq())

        if not os.path.isfile(conf.path.validation):
            logger.warn(
                f"""\
Processing validation batches, queuing {len(self.validation_chunks)} batches."""
            )
            self.qfseq.queue_iterable(self.validation_chunks)
            self._validation_batches = [batch.contiguous() for batch in self.qfseq]
            torch.save(self._validation_batches, conf.path.validation)
            logger.warn("Validation batches pickled.")

        if not os.path.isfile(conf.path.test):
            logger.warn(
                f"""\
Processing testing batches, queuing {len(self.validation_chunks)} batches."""
            )
            self.qfseq.queue_iterable(self.testing_chunks)
            self._testing_batches = [batch.contiguous() for batch in self.qfseq]
            torch.save(self._testing_batches, conf.path.test)
            logger.warn("Testing batches pickled.")

    @property
    def validation_batches(self):
        if not hasattr(self, "_validation_batches"):
            logger.warning("Validation batches not loaded, loading from disk.")
            self._validation_batches = torch.load(
                conf.path.validation, map_location=torch.device("cpu")
            )
            logger.warning("Finished loading.")
        return self._validation_batches

    @property
    def testing_batches(self):
        if not hasattr(self, "_testing_batches"):
            logger.warning("Testing batches not loaded, loading from disk.")
            self._testing_batches = torch.load(
                conf.path.test, map_location=torch.device("cpu")
            )
            logger.warning("Finished loading.")
        return self._testing_batches

    def load_test_batches(self):
        self.testing_batches = torch.load(conf.path.test, map_location=device)

    def queue_epoch(self, n_skip_events=0):
        n_skip_chunks = n_skip_events // conf.loader.chunksize
        # Cycle Epochs
        n_skip_chunks % (len(files) * conf.loader.chunksize)

        n_skip_batches = (
            n_skip_events % conf.loader.chunksize
        ) // conf.loader.batch_size

        if n_skip_events != 0:
            logger.info(
                f"""\
Skipping {n_skip_events} => {n_skip_chunks} chunks and {n_skip_batches} batches."""
            )
        self.epoch_chunks = self.training_chunks[n_skip_chunks:]
        self.qfseq.queue_iterable(self.epoch_chunks)

        if n_skip_batches != 0:
            for _ in range(n_skip_batches):
                _ = next(self.qfseq)
            logger.info(f"Skipped {n_skip_batches} batches.")

    def __iter__(self):
        return iter(self.qfseq)
