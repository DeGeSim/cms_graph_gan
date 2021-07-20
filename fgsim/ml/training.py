import time

import torch
import torch_geometric
from omegaconf import OmegaConf
from tqdm import tqdm

from ..config import conf, device
from ..io.queued_dataset import QueuedDataLoader
from ..monitor import setup_experiment, setup_writer
from ..utils.logger import logger
from ..utils.move_batch_to_device import move_batch_to_device
from .early_stopping import early_stopping
from .holder import model_holder
from .train_state import TrainState
from .validate import validate


def training_step(
    batch: torch_geometric.data.Batch, train_state: TrainState
) -> None:
    train_state.holder.optim.zero_grad()
    output = train_state.holder.model(batch)

    prediction = torch.squeeze(output.T)
    loss = train_state.holder.lossf(prediction, batch[conf.yvar].float())
    loss.backward()
    train_state.state.loss = float(loss)
    train_state.holder.optim.step()


def training_procedure() -> None:
    logger.warn(
        "Starting training with state\n" + OmegaConf.to_yaml(model_holder.state)
    )
    train_state = TrainState(
        model_holder,
        model_holder.state,
        QueuedDataLoader(),
        setup_writer(),
        setup_experiment(model_holder),
    )

    # Check if the training already has finished:
    early_stopping(train_state)

    # Initialize the training
    # switch model in training mode
    train_state.holder.model.train()

    try:
        # Iterate over the Epochs
        for train_state.state.epoch in range(
            train_state.state.epoch, conf.model["n_epochs"]
        ):
            # Iterate over the batches
            train_state.state.batch_start_time = time.time()
            train_state.loader.queue_epoch(
                n_skip_events=train_state.state.processed_events
            ),
            for train_state.state.ibatch, batch in enumerate(
                tqdm(
                    train_state.loader,
                    initial=train_state.state.ibatch,
                ),
                start=train_state.state.ibatch,
            ):
                batch = move_batch_to_device(batch, device)

                train_state.state.time_io_done = time.time()

                training_step(batch, train_state)

                train_state.state.time_training_done = time.time()

                # save the generated torch tensor models to disk
                validate(train_state)

                train_state.state.time_validation_done = time.time()

                early_stopping(train_state)

                train_state.state.time_early_stopping_done = time.time()

                train_state.writelogs()

                # preparatoin for next step
                train_state.state.processed_events += conf.loader.batch_size
                train_state.state["grad_step"] += 1
                train_state.state.batch_start_time = time.time()

            train_state.state.ibatch = 0
            train_state.save_checkpoint()
            train_state.save_best_model()
    except Exception as error:
        logger.error("Error detected, stopping qfseq.")
        train_state.loader.qfseq._stop()
        raise error
