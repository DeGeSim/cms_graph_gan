import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm

from ..config import conf, device
from ..io.queued_dataset import get_loader
from ..utils.logger import logger
from .holder import modelHolder

writer = SummaryWriter(
    f"runs/{conf.log_name}/" + datetime.now().strftime("%Y-%m-%d-%H:%M/")
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
    if holder.state["grad_step"] % 10 == 0:
        losses = []
        for batch in holder.validation_batches:
            batch = batch.to(device)
            prediction = torch.squeeze(holder.model(batch).T)
            losses.append(holder.lossf(prediction, batch.y.float()))
        mean_loss = torch.mean(torch.tensor(losses))
        holder.state.val_losses.append(float(mean_loss))
        writer.add_scalar("val_loss", mean_loss, holder.state["grad_step"])
        mean_loss = float(mean_loss)
        if holder.state.min_val_loss is None or holder.state.min_val_loss > mean_loss:
            holder.state.min_val_loss = mean_loss
            holder.best_grad_step = holder.state["grad_step"]
            holder.best_state_model = holder.model.state_dict()
            holder.best_state_optim = holder.optim.state_dict()

    if holder.state["grad_step"] % 50 == 0:
        holder.save_model()
        holder.save_best_model()


def early_stopping(holder):
    if holder.state["grad_step"] % 10 == 0:
        # the the most recent losses
        # dont stop for the first epochs
        if len(holder.state.val_losses) < conf.training.early_stopping:
            return
        recent_losses = holder.state.val_losses[-conf.training.early_stopping :]
        relative_improvement = 1 - (min(recent_losses) / recent_losses[0])

        if relative_improvement < conf.training.early_stopping_improvement:
            holder.save_model()
            holder.save_best_model()
            logger.warn("Early Stopping criteria fullfilled")
            exit(0)


def training_procedure(holder: modelHolder):
    logger.warn("Starting training with state\n" + OmegaConf.to_yaml(holder.state))
    holder.validation_batches, train_loader = get_loader(holder.state.processed_events)
    # Initialize the training
    holder.model = holder.model.float().to(device)
    holder.model.train()
    holder.state.global_start_time = time.time()
    # Iterate over the Epochs
    for holder.state.epoch in range(holder.state.epoch, conf.model["n_epochs"]):
        # Iterate over the batches
        holder.state.batch_start_time = time.time()
        holder.state.saving_start_time = time.time()
        for holder.state.ibatch, batch in enumerate(
            tqdm(train_loader), start=holder.state.ibatch
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
        holder.save_model()
        holder.save_best_model()
