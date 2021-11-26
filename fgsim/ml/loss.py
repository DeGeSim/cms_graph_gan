"""Dynamically import the losses"""
import importlib
from typing import Callable, Dict

import torch
from omegaconf.dictconfig import DictConfig

from fgsim.config import device
from fgsim.utils.check_for_nans import contains_nans


class SubNetworkLoss:
    def __init__(self, pconf: DictConfig) -> None:
        self.pconf = pconf
        self.parts: Dict[str, Callable] = {}

        for name, lossconf in pconf.items():
            assert name != "parts"
            params = lossconf if lossconf is not None else {}
            if not hasattr(torch.nn, name):
                loss = importlib.import_module(f"fgsim.models.loss.{name}").LossGen(
                    **params
                )
            else:
                loss = getattr(torch.nn, name)().to(device)
            self.parts[name] = loss
            setattr(self, name, loss)

    def __call__(self, holder, ytrue, ypred):
        loss = torch.sum(loss(holder, ytrue, ypred) for loss in self.losses)
        assert not contains_nans(loss)[0]
        return loss


class LossesCol:
    """Holds all losses for a single subnetwork.
    Calling this class should return a single (1D) loss for the gradient step"""

    def __init__(self, pconf: DictConfig) -> None:
        self.pconf = pconf
        self.parts: Dict[str, SubNetworkLoss] = {}

        for name, submodelconf in pconf.items():
            snl = SubNetworkLoss(submodelconf.losses)
            self.parts[name] = snl
            setattr(self, name, snl)
