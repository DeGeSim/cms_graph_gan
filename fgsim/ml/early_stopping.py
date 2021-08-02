from omegaconf import OmegaConf

from ..config import conf
from ..utils.logger import logger
from .train_state import TrainState


def early_stopping(train_state: TrainState) -> bool:
    relative_improvement = 1 - (
        min(train_state.state.val_losses) / train_state.state.min_val_loss
    )

    if relative_improvement < conf.training.early_stopping_improvement:
        train_state.holder.save_checkpoint()
        train_state.writer.flush()
        train_state.writer.close()
        logger.warn("Early Stopping criteria fullfilled")
        OmegaConf.save(train_state.state, conf.path.complete_state)
        train_state.experiment.end()
        return True
    else:
        return False
