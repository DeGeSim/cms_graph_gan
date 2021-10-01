"""Provides the procedure to preprocess the datasets"""

from typing import List

import torch
from torch_geometric.data import Data as GraphType
from tqdm import tqdm

from fgsim.config import conf
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.utils.logger import logger


def preprocess_procedure(
    data_loader: QueuedDataLoader = QueuedDataLoader(),
) -> None:
    logger.warning(
        f"""\
Processing validation batches, queuing {len(data_loader.validation_chunks)} chunks."""
    )
    data_loader.qfseq.queue_iterable(data_loader.validation_chunks)
    validation_batches = [batch for batch in tqdm(data_loader.qfseq)]
    torch.save(validation_batches, conf.path.validation)
    logger.warning("Validation batches pickled.")

    logger.warning(
        f"""\
Processing testing batches, queuing {len(data_loader.testing_chunks)} chunks."""
    )
    data_loader.qfseq.queue_iterable(data_loader.testing_chunks)
    testing_batches = [batch for batch in tqdm(data_loader.qfseq)]
    torch.save(testing_batches, conf.path.test)
    logger.warning("Testing batches pickled.")

    if conf.loader.preprocess_training:
        logger.warning("Processing training batches")
        data_loader.queue_epoch()
        batch_list: List[GraphType] = []
        ifile = 0
        for batch in tqdm(data_loader.qfseq):
            output_file = f"{conf.path.training}/{ifile:03d}.pt"

            if len(batch_list) == data_loader.n_test_batches:
                logger.info(f"Saving {output_file}")
                torch.save(batch_list, f"{output_file}")
                ifile += 1
                batch_list = []
            batch_list.append(batch)
        logger.info(f"Saving {output_file}")
        torch.save(batch_list, f"{output_file}")
    data_loader.qfseq.stop()
    exit(0)
