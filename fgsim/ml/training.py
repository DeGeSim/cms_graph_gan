import sys
import time
from copy import deepcopy

import torch
import torch_geometric
from omegaconf import OmegaConf
from tqdm import tqdm

from ..config import conf, device
from ..io.queued_dataset import QueuedDataLoader
from ..monitor import setup_experiment, setup_writer
from ..utils.check_for_nans import check_chain_for_nans
from ..utils.logger import logger
from ..utils.move_batch_to_device import move_batch_to_device
from .holder import model_holder
from .train_state import TrainState


def training_step(
    batch: torch_geometric.data.Batch, train_state: TrainState
) -> None:
    train_state.holder.optim.zero_grad()
    output = train_state.holder.model(batch)

    prediction = torch.squeeze(output.T)
    loss = train_state.holder.lossf(prediction, batch.y.float())
    loss.backward()
    train_state.state.loss = float(loss)
    train_state.holder.optim.step()


def validate(train_state: TrainState) -> None:
    if train_state.state["grad_step"] % conf.training.validation_interval == 0:
        check_chain_for_nans((train_state.holder.model,))
        losses = []
        for batch in train_state.loader.validation_batches:
            batch = batch.to(device)
            prediction = torch.squeeze(train_state.holder.model(batch).T)
            losses.append(train_state.holder.lossf(prediction, batch.y.float()))

        mean_loss = torch.mean(torch.tensor(losses))
        train_state.state.val_losses.append(float(mean_loss))

        train_state.writer.add_scalar(
            "val_loss", mean_loss, train_state.state["grad_step"]
        )
        train_state.experiment.log_metric(
            "val_loss", mean_loss, train_state.state["grad_step"]
        )

        mean_loss = float(mean_loss)
        if (
            not hasattr(train_state.state, "min_val_loss")
            or train_state.state.min_val_loss > mean_loss
        ):
            train_state.state.min_val_loss = mean_loss
            train_state.state.best_grad_step = train_state.state["grad_step"]
            train_state.holder.best_model_state = deepcopy(
                train_state.holder.model.state_dict()
            )
        assert train_state.state is train_state.holder.state

    if (
        train_state.state["grad_step"] != 0
        and train_state.state["grad_step"] % conf.training.checkpoint_interval == 0
    ):
        train_state.holder.save_models()


def early_stopping(train_state: TrainState) -> None:
    if (
        train_state.state["grad_step"] != 0
        and train_state.state["grad_step"] % conf.training.validation_interval == 0
    ):
        # the the most recent losses
        # dont stop for the first epochs
        if len(train_state.state.val_losses) < conf.training.early_stopping:
            return
        recent_losses = train_state.state.val_losses[
            -conf.training.early_stopping :
        ]
        relative_improvement = 1 - (min(recent_losses) / recent_losses[0])

        if relative_improvement < conf.training.early_stopping_improvement:
            train_state.holder.save_models()
            train_state.writer.flush()
            train_state.writer.close()
            logger.warn("Early Stopping criteria fullfilled")
            if hasattr(train_state, "loader"):
                train_state.loader.qfseq.drain_seq()
            sys.exit()


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
