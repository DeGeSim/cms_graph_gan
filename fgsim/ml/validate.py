from copy import deepcopy

import torch
from tqdm import tqdm

from ..config import conf, device
from ..utils.check_for_nans import check_chain_for_nans
from .train_state import TrainState


def validate(train_state: TrainState) -> None:
    if train_state.state["grad_step"] % conf.training.validation_interval == 0:
        check_chain_for_nans((train_state.holder.model,))
        losses = []
        for batch in tqdm(
            train_state.loader.validation_batches, postfix="validating"
        ):
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
