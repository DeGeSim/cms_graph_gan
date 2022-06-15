from typing import Dict

import numpy as np
from sklearn.linear_model import LinearRegression

from fgsim.config import conf
from fgsim.monitoring.logger import logger


def early_stopping(history: Dict) -> bool:
    """
    If in the last `valsteps` the stopping metric have shown no
    improvement(=decay), return True, else False.

    Args:
      state (DictConfig): DictConfig

    Returns:
      A boolean value.
    """
    # Make sure some metrics have been recorded
    if len(history["val_metrics"].keys()) == 0:
        return False
    # collect at least two values before evaluating the criteria
    if len(history["stop_crit"]) < 2:
        return False
    loss_arrs = []
    for k in conf.models.keys():  # iterate the models
        model_loss_dict = history["losses"][k]
        loss_arrs.append(
            sum(  # sum the losses for each module
                [
                    histdict_to_np(model_loss_dict[lname])
                    for lname in model_loss_dict  # iterate the losses
                ]
            )
        )

    return all(
        [is_minimized(np.array(history["stop_crit"]))]
        + [is_not_dropping(carr) for carr in loss_arrs]
    )


def histdict_to_np(histd):
    if isinstance(histd, dict):
        return np.array(histd["sum"])
    else:
        return np.array(histd)


def is_not_dropping(arr: np.ndarray):
    valsteps = conf.training.early_stopping.validation_steps
    subarr = arr[-valsteps:]
    subm = np.mean(subarr)
    subarr = (subarr - subm) / subm
    reg = LinearRegression()
    reg.fit(
        X=np.arange(len(subarr)).reshape(-1, 1),
        y=subarr.reshape(-1),
    )
    return reg.coef_[-1] > conf.training.early_stopping.improvement


def is_minimized(arr: np.ndarray):
    valsteps = conf.training.early_stopping.validation_steps
    stop_metric = np.array(arr)

    if len(stop_metric) < valsteps + 1:
        return False
    growth = 1 - (min(stop_metric[-valsteps:]) / min(stop_metric[:-valsteps]))

    logger.info(
        f"""Relative Improvement in the last {valsteps} validation steps: {growth}%"""
    )
    if growth < conf.training.early_stopping.improvement:
        return True
    else:
        return False
