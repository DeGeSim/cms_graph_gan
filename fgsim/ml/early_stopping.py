from omegaconf import OmegaConf

from ..config import conf
from ..utils.logger import logger
from .train_state import TrainState


def early_stopping(train_state: TrainState) -> bool:
    if (
        train_state.state["grad_step"] != 0
        and train_state.state["grad_step"] % conf.training.validation_interval == 0
    ):
        # the the most recent losses
        # dont stop for the first epochs
        if len(train_state.state.val_losses) < conf.training.early_stopping:
            return False
        recent_losses = train_state.state.val_losses[
            -conf.training.early_stopping :
        ]
        relative_improvement = 1 - (min(recent_losses) / recent_losses[0])

        if relative_improvement < conf.training.early_stopping_improvement:
            train_state.holder.save_checkpoint()
            train_state.writer.flush()
            train_state.writer.close()
            logger.warn("Early Stopping criteria fullfilled")
            OmegaConf.save(train_state.state, conf.path.complete_state)
            train_state.experiment.end()
            train_state.loader.qfseq.stop()
            return True
    else:
        return False
