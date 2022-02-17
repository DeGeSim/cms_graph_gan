"""
Provides the `QueuedDataLoader` class. The definded sequence of qf steps is \
loaded depending on `conf.loader.name`.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import queueflow as qf
import torch

from fgsim.config import conf
from fgsim.io.preprocessed_seq import preprocessed_seq
from fgsim.io.sel_seq import (
    DataSetType,
    files,
    len_dict,
    postprocess_switch,
    process_seq,
)
from fgsim.monitoring.logger import logger

chunksize = conf.loader.chunksize
batch_size = conf.loader.batch_size

ChunkType = List[Tuple[Path, int, int]]


class QueuedDataLoader:
    """
`QueuedDataLoader` makes `validation_batches` \
and `testing_batches` available as properties; to load training batches, one \
must queue an epoch via `queue_epoch()` and iterate over the instance of the class.
    """

    def __init__(self):
        chunk_coords = self._compute_chucks()

        np.random.shuffle(chunk_coords)

        # Make sure the chunks can be split evenly into batches:
        assert chunksize % batch_size == 0

        assert conf.loader.validation_set_size % chunksize == 0
        n_validation_batches = conf.loader.validation_set_size // batch_size
        n_validation_chunks = conf.loader.validation_set_size // chunksize

        assert conf.loader.test_set_size % chunksize == 0
        self.n_test_batches = conf.loader.test_set_size // batch_size
        n_testing_chunks = conf.loader.test_set_size // chunksize

        logger.info(
            f"Using the first {n_validation_batches} batches for "
            + f"validation and the next {self.n_test_batches} batches for testing."
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

        # Get access to the postprocess switch for computing the validation dataset
        self.postprocess_switch = postprocess_switch

        if conf.command != "preprocess" and conf.loader.preprocess_training:
            qf.init(False)
            self.qfseq = qf.Sequence(*preprocessed_seq())
        else:
            qf.init(False)
            self.qfseq = qf.Sequence(*process_seq())

        if conf.command != "preprocess":
            # In all cases training and test set must be available
            # if the current command is not  preprocessing
            # if (
            #     not os.path.isfile(conf.path.validation)
            #     or not os.path.isfile(conf.path.test)
            # ):
            #     raise FileNotFoundError
            if conf.loader.preprocess_training:
                self.preprocessed_files = list(
                    sorted(Path(conf.path.training).glob(conf.path.training_glob))
                )
                if len(self.preprocessed_files) == 0:
                    raise FileNotFoundError

    def _compute_chucks(self) -> List[ChunkType]:
        chunk_coords: List[ChunkType] = [[]]
        ifile = 0
        ielement = 0
        current_chunck_elements = 0
        while ifile < len(files):
            elem_left_in_cur_file = len_dict[str(files[ifile])] - ielement
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
        return chunk_coords

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

        n_skip_epochs = n_skip_events // (
            conf.loader.chunksize * len(self.training_chunks)
        )

        # Compute the batches on the fly
        if not conf.loader.preprocess_training or conf.command == "preprocess":
            # Repeat the shuffeling to get the same list
            for _ in range(n_skip_epochs):
                np.random.shuffle(self.training_chunks)

            # Cycle Epochs
            n_skip_chunks = (n_skip_events // conf.loader.chunksize) % len(
                self.training_chunks
            )
            # Only queue to the chucks that are still left
            epoch_chunks = self.training_chunks[n_skip_chunks:]
            self.qfseq.queue_iterable(epoch_chunks)
            np.random.shuffle(self.training_chunks)

            # No calculate the number of batches that we still have to skip,
            # because a chunk may be multiple batches and we need to skip
            # the ones that are alread processed
            n_skip_batches = (
                n_skip_events % conf.loader.chunksize
            ) // conf.loader.batch_size

            logger.info(
                f"""\
Skipping {n_skip_events} events => {n_skip_chunks} chunks and {n_skip_batches} batches."""
            )

            for _ in range(n_skip_batches):
                _ = next(self.qfseq)

        # Load the preprocessed batches
        else:
            # Repeat the shuffeling to get the same list
            for _ in range(n_skip_epochs):
                np.random.shuffle(self.preprocessed_files)

            # Calculate the number of files that have already been processed
            # one file contains self.n_test_batches batches
            n_skip_files = (
                (n_skip_events // conf.loader.batch_size)  # n batches
                // self.n_test_batches  # by the number of batches per file
                % len(self.preprocessed_files)  # modulo the files per epoch
            )
            epoch_files = self.preprocessed_files[n_skip_files:]
            self.qfseq.queue_iterable(epoch_files)
            np.random.shuffle(self.preprocessed_files)

            # No calculate the number of batches that we still have to skip
            n_skip_batches = (
                (n_skip_events // conf.loader.batch_size)  # n batches
            ) % self.n_test_batches  # modulo the batches in a file

            logger.info(
                f"""\
    Skipping {n_skip_events} events => {n_skip_files} files and {n_skip_batches} batches."""
            )

            # Skip the correct number of batches.
            for ibatch in range(n_skip_batches):
                _ = next(self.qfseq)
                logger.debug(f"Skipped batch({ibatch}).")

    def __iter__(self) -> qf.Sequence:
        return iter(self.qfseq)
