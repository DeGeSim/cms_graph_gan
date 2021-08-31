"""
Provides the `QueuedDataLoader` class. The definded sequence of qf steps is \
loaded depending on `conf.loader.name`.
"""

import importlib
import os
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import yaml

from ..config import conf
from ..utils.logger import logger
from . import qf

# Import the specified processing sequence
process_seq = importlib.import_module(
    f"..io.{conf.loader.name}", "fgsim.models"
).process_seq


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
    """
`QueuedDataLoader` makes `validation_batches` \
and `testing_batches` available as properties; to load training batches, one \
must queue an epoch via `queue_epoch()` and iterate over the instance of the class.
    """

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
Processing validation batches, queuing {len(self.validation_chunks)} chunks."""
            )
            self.qfseq.queue_iterable(self.validation_chunks)
            self._validation_batches = [batch for batch in self.qfseq]
            torch.save(self._validation_batches, conf.path.validation)
            logger.warn("Validation batches pickled.")

        if not os.path.isfile(conf.path.test):
            logger.warn(
                f"""\
Processing testing batches, queuing {len(self.validation_chunks)} chunks."""
            )
            self.qfseq.queue_iterable(self.testing_chunks)
            self._testing_batches = [batch for batch in self.qfseq]
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

    def queue_epoch(self, n_skip_events=0):
        n_skip_chunks = n_skip_events // conf.loader.chunksize
        # Cycle Epochs
        n_skip_chunks = n_skip_chunks % len(self.training_chunks)

        n_skip_batches = (
            n_skip_events % conf.loader.chunksize
        ) // conf.loader.batch_size

        if n_skip_events != 0:
            logger.info(
                f"""\
Skipping {n_skip_events} events => {n_skip_chunks} chunks and {n_skip_batches} batches."""
            )
        epoch_chunks = self.training_chunks[n_skip_chunks:]
        self.qfseq.queue_iterable(epoch_chunks)

        if n_skip_batches != 0:
            for _ in range(n_skip_batches):
                _ = next(self.qfseq)
            logger.info(f"Skipped {n_skip_batches} batches.")

    def __iter__(self):
        return iter(self.qfseq)
