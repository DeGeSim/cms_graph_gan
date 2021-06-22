import sys
import time

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

from ..config import conf, device
from ..io.queued_dataset import QueuedDataLoader
from ..utils.logger import logger
from .holder import model_holder as holder

writer = SummaryWriter(conf.path.tensorboard)


def writelogs():
    writer.add_scalars(
        "times",
        {
            "batch_start_time": holder.state.batch_start_time
            - holder.state.global_start_time,
            "model_start_time": holder.state.model_start_time
            - holder.state.global_start_time,
            "batchtotal": holder.state.saving_start_time
            - holder.state.global_start_time,
        },
        holder.state["grad_step"],
    )

    writer.add_scalar("loss", holder.loss, holder.state["grad_step"])

    writer.flush()


def training_step(batch):
    holder.optim.zero_grad()
    prediction = torch.squeeze(holder.model(batch).T)
    holder.loss = holder.lossf(prediction, batch.y.float())
    holder.loss.backward()
    holder.optim.step()


def validate():
    if (
        holder.state["grad_step"] != 0
        and holder.state["grad_step"] % conf.training.validation_interval == 0
    ):
        losses = []
        for batch in holder.loader.validation_batches:
            batch = batch.to(device)
            prediction = torch.squeeze(holder.model(batch).T)
            losses.append(holder.lossf(prediction, batch.y.float()))
        mean_loss = torch.mean(torch.tensor(losses))
        holder.state.val_losses.append(float(mean_loss))
        writer.add_scalar("val_loss", mean_loss, holder.state["grad_step"])
        mean_loss = float(mean_loss)
        if (
            not hasattr(holder.state, "min_val_loss")
            or holder.state.min_val_loss > mean_loss
        ):
            holder.state.min_val_loss = mean_loss
            holder.best_grad_step = holder.state["grad_step"]
            holder.best_model_state = deepcopy(holder.model.state_dict())

    if (
        holder.state["grad_step"] != 0
        and holder.state["grad_step"] % conf.training.checkpoint_interval == 0
    ):
        holder.save_models()


def early_stopping():
    if (
        holder.state["grad_step"] != 0
        and holder.state["grad_step"] % conf.training.validation_interval == 0
    ):
        # the the most recent losses
        # dont stop for the first epochs
        if len(holder.state.val_losses) < conf.training.early_stopping:
            return
        recent_losses = holder.state.val_losses[-conf.training.early_stopping :]
        relative_improvement = 1 - (min(recent_losses) / recent_losses[0])

        if relative_improvement < conf.training.early_stopping_improvement:
            holder.save_models()
            writer.flush()
            writer.close()
            logger.warn("Early Stopping criteria fullfilled")
            holder.loader.qfseq.drain_seq()
            sys.exit()


def training_procedure():
    logger.warn("Starting training with state\n" + OmegaConf.to_yaml(holder.state))
    # Check if the training already has finished:
    early_stopping()

    # Initialize the training
    # switch model in training mode
    holder.model.train()
    holder.state.global_start_time = time.time()
    holder.loader = QueuedDataLoader()
    # Iterate over the Epochs
    for holder.state.epoch in range(holder.state.epoch, conf.model["n_epochs"]):
        # Iterate over the batches
        holder.state.batch_start_time = time.time()
        holder.state.saving_start_time = time.time()
        for holder.state.ibatch, batch in enumerate(
            tqdm(
                holder.loader.get_epoch_generator(
                    n_skip_events=holder.state.processed_events
                ),
                initial=holder.state.ibatch,
            ),
            start=holder.state.ibatch,
        ):
            holder.state.model_start_time = time.time()

            training_step(batch)

            holder.state.saving_start_time = time.time()

            # save the generated torch tensor models to disk
            validate()

            writelogs()

            early_stopping()

            # preparatoin for next step
            holder.state.processed_events += conf.loader.batch_size
            holder.state["grad_step"] += 1
            holder.state.batch_start_time = time.time()

        holder.state.ibatch = 0
        holder.save_checkpoint()
        holder.save_best_model()
