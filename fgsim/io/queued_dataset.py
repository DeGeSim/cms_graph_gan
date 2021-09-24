"""
Provides the `QueuedDataLoader` class. The definded sequence of qf steps is \
loaded depending on `conf.loader.name`.
"""

import importlib
import os

import numpy as np
import torch

from fgsim.config import conf
from fgsim.geo.batchtype import DataSetType
from fgsim.io import qf
from fgsim.io.qf.sequence import Sequence as qfseq
from fgsim.utils.logger import logger

# Import the specified processing sequence
sel_seq = importlib.import_module(f"fgsim.io.{conf.loader.name}", "fgsim.models")

process_seq = sel_seq.process_seq
files = sel_seq.files
len_dict = sel_seq.len_dict


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
                    (files[ifile], ielement, ielement + elem_to_add)
                )
                ielement += elem_to_add
                current_chunck_elements += elem_to_add
            else:
                chunk_coords[-1].append(
                    (files[ifile], ielement, ielement + elem_left_in_cur_file)
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

        # Check that there is a reasonable amount of data
        assert (
            len(self.validation_chunks) + len(self.testing_chunks)
            < len(self.training_chunks) / 2
        ), "Dataset to small"

        # Assign the sequence with the specifice steps needed to process the dataset.
        self.qfseq = qf.Sequence(*process_seq())

        if not os.path.isfile(conf.path.validation):
            logger.warning(
                f"""\
Processing validation batches, queuing {len(self.validation_chunks)} chunks."""
            )
            self.qfseq.queue_iterable(self.validation_chunks)
            self._validation_batches = [batch for batch in self.qfseq]
            torch.save(self._validation_batches, conf.path.validation)
            logger.warning("Validation batches pickled.")

        if not os.path.isfile(conf.path.test):
            logger.warning(
                f"""\
Processing testing batches, queuing {len(self.validation_chunks)} chunks."""
            )
            self.qfseq.queue_iterable(self.testing_chunks)
            self._testing_batches = [batch for batch in self.qfseq]
            torch.save(self._testing_batches, conf.path.test)
            logger.warning("Testing batches pickled.")

    @property
    def validation_batches(self) -> DataSetType:
        if not hasattr(self, "_validation_batches"):
            logger.warning("Validation batches not loaded, loading from disk.")
            self._validation_batches = torch.load(
                conf.path.validation, map_location=torch.device("cpu")
            )
            logger.warning(
                f"Finished loading. Type is{type(self._validation_batches)}"
            )
        return self._validation_batches

    @property
    def testing_batches(self) -> DataSetType:
        if not hasattr(self, "_testing_batches"):
            logger.warning("Testing batches not loaded, loading from disk.")
            self._testing_batches = torch.load(
                conf.path.test, map_location=torch.device("cpu")
            )
            logger.warning("Finished loading.")
        return self._testing_batches

    def queue_epoch(self, n_skip_events=0) -> None:
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

    def __iter__(self) -> qfseq:
        return iter(self.qfseq)
