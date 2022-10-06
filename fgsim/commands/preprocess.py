"""Provides the procedure to preprocess the datasets"""

from typing import List

import torch
from torch_geometric.data import Data as GraphType
from tqdm import tqdm

from fgsim.config import conf
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import loader_info
from fgsim.monitoring.logger import logger


def preprocess_procedure() -> None:
    loader_info.file_manager.save_len_dict()
    loader_info.scaler.save_scaler()
    data_loader = QueuedDataset(loader_info)
    data_loader.qfseq.start()

    logger.warning(
        f"""\
Processing validation batches, queuing {len(data_loader.validation_chunks)} chunks."""
    )
    # Turn the postprocessing off for the validation and testing
    data_loader.shared_postprocess_switch.value = 1

    # Validation Batches
    data_loader.shared_batch_size.value = conf.loader.batch_size

    data_loader.qfseq.queue_iterable(data_loader.validation_chunks)
    validation_batch = [e for e in data_loader.qfseq]
    torch.save(validation_batch, conf.path.validation)
    logger.warning(f"Validation batches pickled to {conf.path.validation}.")

    logger.warning(
        f"""\
Processing testing batches, queuing {len(data_loader.testing_chunks)} chunks."""
    )
    # Test Batch
    # data_loader.shared_batch_size.value = conf.loader.test_set_size
    data_loader.qfseq.queue_iterable(data_loader.testing_chunks)
    test_batch = [e for e in data_loader.qfseq]
    torch.save(test_batch, conf.path.test)
    logger.warning(f"Testing batches pickled to {conf.path.test}.")

    # if conf.loader.preprocess_training:
    logger.warning("Processing training batches")
    # Turn the postprocessing off for the training
    data_loader.shared_postprocess_switch.value = 0
    data_loader.shared_batch_size.value = conf.loader.batch_size
    data_loader.queue_epoch()
    batch_list: List[GraphType] = []
    ifile = 0
    for batch in tqdm(data_loader.qfseq):
        output_file = f"{conf.path.training}/{ifile:03d}.pt"
        batch_list.append(batch)
        if len(batch_list) == conf.loader.events_per_file // conf.loader.batch_size:
            logger.info(f"Saving {output_file}")
            torch.save(batch_list, f"{output_file}")
            ifile += 1
            batch_list = []
    logger.info(f"Saving {output_file}")
    torch.save(batch_list, f"{output_file}")
    data_loader.qfseq.stop()
    exit(0)
