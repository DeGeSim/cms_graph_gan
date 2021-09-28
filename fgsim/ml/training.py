import time
from typing import Dict, Union

import torch
import torch_geometric
from omegaconf import OmegaConf
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.monitor import setup_experiment, setup_writer
from fgsim.utils.batch_utils import move_batch_to_device
from fgsim.utils.logger import logger

from .early_stopping import early_stopping
from .holder import model_holder
from .train_state import TrainState
from .validate import validate


def training_step(
    batch: Union[torch_geometric.data.Batch, Dict[str, torch.Tensor]],
    train_state: TrainState,
) -> None:
    train_state.holder.optim.zero_grad()
    output = train_state.holder.model(batch)

    prediction = torch.squeeze(output.T)
    # Check for the global_add_pool bug in pytorch_geometric
    # https://github.com/rusty1s/pytorch_geometric/issues/2895
    if len(prediction) != len(batch.y):
        return
    loss = train_state.holder.lossf(y=batch.y, yhat=prediction)
    loss.backward()
    train_state.holder.optim.step()

    train_state.state.loss = float(loss)


def training_procedure() -> None:
    logger.warning(
        "Starting training with state\n" + OmegaConf.to_yaml(model_holder.state)
    )
    train_state = TrainState(
        model_holder,
        model_holder.state,
        QueuedDataLoader(),
        setup_writer(),
        setup_experiment(model_holder) if not conf.debug else None,
    )

    # Initialize the training

    # Queue that batches
    train_state.loader.queue_epoch(n_skip_events=train_state.state.processed_events)
    if not conf.debug and train_state.experiment.ended:
        logger.warning("Training has been completed, stopping.")
        train_state.loader.qfseq.stop()
        exit(0)
    try:
        while not early_stopping(train_state):
            # switch model in training mode
            train_state.holder.model.train()
            for _ in tqdm(
                range(conf.training.validation_interval), postfix="training"
            ):
                train_state.state.batch_start_time = time.time()
                try:
                    batch = next(train_state.loader.qfseq)
                except StopIteration:
                    # If there is no next batch go to the next epoch
                    if not conf.debug:
                        train_state.experiment.log_epoch_end(
                            train_state.state["epoch"],
                            step=train_state.state["grad_step"],
                        )
                    logger.warning("New epoch!")
                    train_state.state.epoch += 1
                    train_state.state.ibatch = 0
                    train_state.loader.queue_epoch(
                        n_skip_events=train_state.state.processed_events
                    )
                    batch = next(train_state.loader.qfseq)
                batch = move_batch_to_device(batch, device)
                train_state.state.time_io_done = time.time()
                training_step(batch, train_state)
                train_state.state.time_training_done = time.time()
                train_state.write_trainstep_logs(batch)
                train_state.state.ibatch += 1
                train_state.state.processed_events += conf.loader.batch_size
                train_state.state["grad_step"] += 1

            validate(train_state)
            train_state.holder.save_checkpoint()
    except Exception as error:
        logger.error("Error detected, stopping qfseq.")
        train_state.loader.qfseq.stop()
        raise error
    train_state.loader.qfseq.stop()
    train_state.experiment.end()
    exit(0)
