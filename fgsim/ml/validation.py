from copy import deepcopy

import torch
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.io.sel_seq import batch_tools
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
            holder.gen_points = batch_tools.batch_compute_hlvs(holder.gen_points)
            holder.val_loss(holder, batch)
    logger.warning("Validation batches evaluated")
    holder.val_loss.log_losses(holder.state)
    logger.warning("Validation loss logged")

    min_stop_crit = min(holder.state.stop_crit)
    if min_stop_crit == holder.state.stop_crit[-1]:
        holder.state.best_grad_step = holder.state["grad_step"]
        holder.best_model_state = deepcopy(holder.models.state_dict())

        if not conf.debug:
            holder.train_log.experiment.log_metric("min_stop_crit", min_stop_crit)
            holder.train_log.experiment.log_metric(
                "best_grad_step", holder.state["grad_step"]
            )
            holder.train_log.experiment.log_metric(
                "best_grad_epoch", holder.state["epoch"]
            )
    logger.warning("Validation function done")
