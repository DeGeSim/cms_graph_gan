from copy import deepcopy

import torch
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.utils.check_for_nans import check_chain_for_nans


def validate(holder: Holder, loader: QueuedDataset) -> None:
    holder.models.eval()
    check_chain_for_nans((holder.models,))
    # Make sure the batches are loaded
    _ = loader.validation_batches
    # Iterate over the validation sample
    for sim_batch in tqdm(loader.validation_batches, postfix="validating"):
        with torch.no_grad():
            sim_batch = sim_batch.clone().to(device)
            holder.reset_gen_points()
            D_sim = holder.models.disc(sim_batch)
            D_gen = holder.models.disc(holder.gen_points)
            holder.val_loss(holder.gen_points, sim_batch, D_sim, D_gen)
    holder.val_loss.log_metrics()

    min_stop_crit = min(holder.history["stop_crit"])
    if min_stop_crit == holder.history["stop_crit"][-1]:
        holder.state.best_grad_step = holder.state["grad_step"]
        holder.best_model_state = deepcopy(holder.models.state_dict())

        if not conf.debug:
            holder.train_log.experiment.log_metric("min_stop_crit", min_stop_crit)
            holder.train_log.experiment.log_metric(
                "best_grad_step", holder.state["grad_step"]
            )
            holder.train_log.experiment.log_metric(
                "best_epoch", holder.state["epoch"]
            )
    logger.warning("Validation function done")
    if not conf.debug:
        holder.train_log.writer.flush()
