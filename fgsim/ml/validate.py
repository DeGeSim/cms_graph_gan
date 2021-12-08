from copy import deepcopy

import torch
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.utils.check_for_nans import check_chain_for_nans


def validate(holder: Holder, loader: QueuedDataLoader) -> None:
    holder.models.eval()
    check_chain_for_nans((holder.models,))
    # Make sure the batches are loaded
    _ = loader.validation_batches
    # Iterate over the validation sample
    for batch in tqdm(loader.validation_batches, postfix="validating"):
        with torch.no_grad():
            batch = batch.clone().to(device)
            holder.reset_gen_points()
            holder.postprocess_gen_points()
            holder.val_loss(holder, batch)
    holder.val_loss.log_losses(holder.state)
    if min(holder.state.val_loss_sum) == holder.state.val_loss_sum[-1]:
        holder.best_model_state = holder.models.state_dict()


def log_validation(holder: Holder, train_log: TrainLog):
    val_loss: float = holder.state.val_losses[-1]
    logger.info(f"Validation Loss: {val_loss}")

    if not conf.debug:
        train_log.writer.add_scalar("val_loss", val_loss, holder.state["grad_step"])
        train_log.experiment.log_metric(
            "val_loss", val_loss, holder.state["grad_step"]
        )
    if (
        not hasattr(holder.state, "min_val_loss")
        or holder.state.min_val_loss > val_loss
    ):
        holder.state.min_val_loss = val_loss
        holder.state.best_grad_step = holder.state["grad_step"]
        holder.best_model_state = deepcopy(holder.model.state_dict())

        if not conf.debug:
            train_log.experiment.log_metric("min_val_loss", val_loss)
            train_log.experiment.log_metric(
                "best_grad_step", holder.state["grad_step"]
            )
            train_log.experiment.log_metric(
                "best_grad_epoch", holder.state["epoch"]
            )
            train_log.experiment.log_metric(
                "best_grad_batch", holder.state["ibatch"]
            )
