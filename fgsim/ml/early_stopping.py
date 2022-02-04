import numpy as np
from omegaconf.dictconfig import DictConfig

from fgsim.config import conf
from fgsim.monitoring.logger import logger


def early_stopping(state: DictConfig) -> bool:
    """
    If in the last `valsteps` the stopping metric have shown no
    improvement(=decay), return True, else False.

    Args:
      state (DictConfig): DictConfig

    Returns:
      A boolean value.
    """
    # Make sure some metrics have been recorded
    if len(state.val_metrics.keys()) == 0:
        return False

    valsteps = conf.training.early_stopping.validation_steps
    stop_metric = np.array(state.stop_crit)

    if len(stop_metric) < valsteps + 1:
        return False
    growth = 1 - (min(stop_metric[-valsteps:]) / min(stop_metric[:-valsteps]))

    logger.info(
        f"""\
Relative Improvement in the last {valsteps} \
validation steps: {growth}%"""
    )
    if growth < conf.training.early_stopping.improvement:
        return True
    else:
        return False
