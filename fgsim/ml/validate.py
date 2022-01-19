from copy import deepcopy

import torch
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
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
            holder.gen_points.compute_hlvs()
            holder.val_loss(holder, batch)
    logger.warning("Validation batches evaluated")
    holder.val_loss.log_losses(holder.state)
    logger.warning("Validation loss logged")
    val_loss_sum = holder.state.val_losses[conf.training.val_loss_sumkey]
    min_val_loss_sum = min(val_loss_sum)
    if min_val_loss_sum == val_loss_sum[-1]:
        holder.state.best_grad_step = holder.state["grad_step"]
        holder.best_model_state = deepcopy(holder.model.state_dict())

        if not conf.debug:
            holder.train_log.experiment.log_metric("min_val_loss", min_val_loss_sum)
            holder.train_log.experiment.log_metric(
                "best_grad_step", holder.state["grad_step"]
            )
            holder.train_log.experiment.log_metric(
                "best_grad_epoch", holder.state["epoch"]
            )
            holder.train_log.experiment.log_metric(
                "best_grad_batch", holder.state["ibatch"]
            )
    logger.warning("Validation function done")
