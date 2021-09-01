from omegaconf import OmegaConf

from fgsim.config import conf
from fgsim.utils.logger import logger

from .train_state import TrainState


def early_stopping(train_state: TrainState) -> bool:
    """Compare the last `conf.training.early_stopping.validation_steps`
    validation losses with the validation losses before that.
    If the minimum has not been reduced by
    `conf.training.early_stopping.improvement`, stop the training"""
    valsteps = conf.training.early_stopping.validation_steps
    if len(train_state.state.val_losses) < valsteps + 1:
        return False
    relative_improvement = 1 - (
        min(train_state.state.val_losses[-valsteps:])
        / min(train_state.state.val_losses[:-valsteps])
    )

    logger.info(
        f"""\
Relative Improvement in the last {valsteps} \
validation steps: {relative_improvement*100}%"""
    )
    if relative_improvement < conf.training.early_stopping.improvement:
        train_state.holder.save_checkpoint()
        train_state.writer.flush()
        train_state.writer.close()
        logger.warn("Early Stopping criteria fulfilled")
        OmegaConf.save(train_state.state, conf.path.complete_state)
        if not conf.debug:
            train_state.experiment.log_other("ended", True)
            train_state.experiment.end()
        return True
    else:
        train_state.experiment.log_other("ended", False)
        return False
