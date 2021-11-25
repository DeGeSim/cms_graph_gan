from copy import deepcopy

import torch
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.utils.batch_utils import move_batch_to_device
from fgsim.utils.check_for_nans import check_chain_for_nans, is_anormal_tensor


def validate(holder: Holder, loader: QueuedDataLoader) -> None:
    holder.model.eval()
    check_chain_for_nans((holder.model,))
    losses = []
    # Make sure the batches are loaded
    _ = loader.validation_batches
    # Iterate over the validation sample
    for batch in tqdm(loader.validation_batches, postfix="validating"):
        batch_gpu = move_batch_to_device(batch, device)
        with torch.no_grad():
            prediction = torch.squeeze(holder.model(batch_gpu).T)
            loss = holder.lossf(ytrue=batch_gpu.ytrue, ypred=prediction)
        if is_anormal_tensor(loss):
            raise ValueError
        losses.append(loss)
        del batch_gpu

    mean_loss = torch.mean(torch.tensor(losses))

    holder.state.val_losses.append(float(mean_loss))


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
