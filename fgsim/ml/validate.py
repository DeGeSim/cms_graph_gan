from copy import deepcopy

import torch
from tqdm import tqdm

from ..config import conf, device
from ..utils.batch_utils import move_batch_to_device
from ..utils.check_for_nans import check_chain_for_nans
from ..utils.logger import logger
from .train_state import TrainState


def validate(train_state: TrainState) -> None:
    train_state.holder.model.eval()
    check_chain_for_nans((train_state.holder.model,))
    losses = []
    # Make sure the batches are loaded
    _ = train_state.loader.validation_batches
    for batch in tqdm(train_state.loader.validation_batches, postfix="validating"):
        batch_gpu = move_batch_to_device(batch, device)
        with torch.no_grad():
            prediction = torch.squeeze(train_state.holder.model(batch_gpu).T)
            loss = train_state.holder.lossf(y=batch_gpu[conf.yvar], yhat=prediction)
        losses.append(loss)
        del batch_gpu

    mean_loss = torch.mean(torch.tensor(losses))

    logger.info(f"Validation Loss: {mean_loss}")
    train_state.state.val_losses.append(float(mean_loss))

    if not conf.debug:
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

        if not conf.debug:
            train_state.experiment.log_metric("min_val_loss", mean_loss)
            train_state.experiment.log_metric(
                "best_grad_step", train_state.state["grad_step"]
            )
            train_state.experiment.log_metric(
                "best_grad_epoch", train_state.state["epoch"]
            )
            train_state.experiment.log_metric(
                "best_grad_batch", train_state.state["ibatch"]
            )
