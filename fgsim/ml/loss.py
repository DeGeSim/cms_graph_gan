"""Dynamically import the losses"""
import importlib
from typing import Callable, List

import torch
from omegaconf import OmegaConf

from fgsim.config import device
from fgsim.utils.check_for_nans import contains_nans


class Loss:
    "Manages a single loss."

    def __init__(self, pconf: OmegaConf) -> None:
        self.pconf = pconf
        self.lossf: Callable
        if not hasattr(torch.nn, self.pconf.name):
            self.lossf = importlib.import_module(
                f"fgsim.models.{self.pconf}"
            ).LossGen(**self.pconf.param)
        else:
            self.lossf = getattr(torch.nn, self.pconf)().to(device)

        self.factor: torch.Tensor = torch.tensor(self.pconf.factor)

    def __call__(self, *args, **kwargs):
        return self.lossf(*args, **kwargs)


class LossesCol:
    """Holds all losses for a single subnetwork.
    Calling this class should return a single (1D) loss for the gradient step"""

    def __init__(self, pconf: OmegaConf) -> None:
        self.pconf = pconf
        self.losses: List[Callable] = [Loss(e) for e in pconf]

    def __call__(self, ytrue, ypred):
        loss = torch.sum(loss(ytrue, ypred) for loss in self.losses)
        assert not contains_nans(loss)[0]
        return loss
