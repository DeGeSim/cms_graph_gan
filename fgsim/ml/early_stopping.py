from omegaconf import OmegaConf

from fgsim.config import conf
from fgsim.utils.logger import logger


def early_stopping(state: OmegaConf) -> bool:
    """Compare the last `conf.training.early_stopping.validation_steps`
    validation losses with the validation losses before that.
    If the minimum has not been reduced by
    `conf.training.early_stopping.improvement`, stop the training"""
    valsteps = conf.training.early_stopping.validation_steps
    if len(state.val_losses) < valsteps + 1:
        return False
    relative_improvement = 1 - (
        min(state.val_losses[-valsteps:]) / min(state.val_losses[:-valsteps])
    )

    logger.info(
        f"""\
Relative Improvement in the last {valsteps} \
validation steps: {relative_improvement*100}%"""
    )
    if relative_improvement < conf.training.early_stopping.improvement:
        return True
    else:
        return False
