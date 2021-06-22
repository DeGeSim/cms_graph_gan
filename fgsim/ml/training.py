import sys
import time

import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import conf, device
from ..io.queued_dataset import QueuedDataLoader
from ..utils.logger import logger
from .holder import modelHolder

writer = SummaryWriter(
    f"runs/{conf.tag}"  # + datetime.now().strftime("%Y-%m-%d-%H:%M/")
)


def writelogs(holder):
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


def training_step(holder, batch):
    holder.optim.zero_grad()
    prediction = torch.squeeze(holder.model(batch).T)
    holder.loss = holder.lossf(prediction, batch.y.float())
    holder.loss.backward()
    holder.optim.step()


def validate(holder):
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
            holder.best_state_model = holder.model.state_dict()
            holder.best_state_optim = holder.optim.state_dict()

    if holder.state["grad_step"] % 50 == 0:
        holder.save_checkpoint()


def early_stopping(holder):
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
            holder.save_best_model()
            writer.flush()
            writer.close()
            logger.warn("Early Stopping criteria fullfilled")
            holder.loader.qfseq.drain_seq()
            sys.exit()


def training_procedure(holder: modelHolder):
    logger.warn("Starting training with state\n" + OmegaConf.to_yaml(holder.state))
    early_stopping(holder)

    # Initialize the training
    holder.model = holder.model.float().to(device)
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
                )
            ),
            start=holder.state.ibatch,
        ):
            holder.state.model_start_time = time.time()

            training_step(holder, batch)

            holder.state.saving_start_time = time.time()

            # save the generated torch tensor models to disk
            validate(holder)

            writelogs(holder)

            early_stopping(holder)

            # preparatoin for next step
            holder.state.processed_events += conf.loader.batch_size
            holder.state["grad_step"] += 1
            holder.state.batch_start_time = time.time()

        holder.state.ibatch = 0
        holder.save_checkpoint()
        holder.save_best_model()
