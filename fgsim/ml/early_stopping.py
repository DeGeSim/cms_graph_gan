from omegaconf.dictconfig import DictConfig

from fgsim.config import conf
from fgsim.monitoring.logger import logger


def early_stopping(state: DictConfig) -> bool:
    """Compare the last `conf.training.early_stopping.validation_steps`
    validation losses with the validation losses before that.
    If the minimum has not been reduced by
    `conf.training.early_stopping.improvement`, stop the training"""
    if len(state.val_losses.keys()) == 0:
        return False
    valsteps = conf.training.early_stopping.validation_steps
    val_loss_sum = state.val_losses[conf.training.val_loss_sumkey]
    if len(val_loss_sum) < valsteps + 1:
        return False
    relative_improvement = 1 - (
        min(val_loss_sum[-valsteps:]) / min(val_loss_sum[:-valsteps])
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
