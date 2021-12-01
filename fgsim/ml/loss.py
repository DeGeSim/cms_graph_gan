"""Dynamically import the losses"""
import importlib
from typing import Callable, Dict

import torch
from omegaconf.dictconfig import DictConfig

from fgsim.config import conf, device
from fgsim.io.queued_dataset import BatchType
from fgsim.monitoring.train_log import TrainLog
from fgsim.utils.check_for_nans import contains_nans


class SubNetworkLoss:
    """Holds all losses for a single subnetwork.
    Calling this class should return a single (1D) loss for the gradient step"""

    def __init__(
        self, subnetworkname: str, pconf: DictConfig, train_logger: TrainLog
    ) -> None:
        self.name = subnetworkname
        self.pconf = pconf
        self.train_logger = train_logger
        self.parts: Dict[str, Callable] = {}

        for lossname, lossconf in pconf.items():
            assert lossname != "parts"
            params = lossconf if lossconf is not None else {}
            if not hasattr(torch.nn, lossname):
                loss = importlib.import_module(
                    f"fgsim.models.loss.{lossname}"
                ).LossGen(**params)
            else:
                loss = getattr(torch.nn, lossname)().to(device)
            self.parts[lossname] = loss
            setattr(self, lossname, loss)

    def __call__(self, holder, batch: BatchType):
        lossesdict = {
            lossname: loss(holder, batch) for lossname, loss in self.parts.items()
        }
        # write the loss to state so it can be logged later
        for lossname, loss in lossesdict.items():
            self.train_logger.log_loss(f"loss.{self.name}.{lossname}", float(loss))
            holder.state.losses[self.name][lossname].append(float(loss))
        summedloss = sum([e for e in lossesdict.values()])
        assert not contains_nans(summedloss)[0]
        return summedloss

    def __getitem__(self, lossname: str) -> Callable:
        return self.parts[lossname]


class LossesCol:
    """Holds all losses for all subnetworks as attributes or as a dict."""

    def __init__(self, train_logger: TrainLog) -> None:
        self.parts: Dict[str, SubNetworkLoss] = {}

        for subnetworkname, subnetworkconf in conf.models.items():
            snl = SubNetworkLoss(
                subnetworkname, subnetworkconf.losses, train_logger
            )
            self.parts[subnetworkname] = snl
            setattr(self, subnetworkname, snl)

    def __getitem__(self, subnetworkname: str) -> SubNetworkLoss:
        return self.parts[subnetworkname]


class ValidationLoss:
    """Holds all losses for a single subnetwork.
    Calling this class should return a single (1D) loss for the gradient step"""

    def __init__(self, train_logger: TrainLog) -> None:
        self.name = "val_loss"
        self.train_logger = train_logger
        self.parts: Dict[str, Callable] = {}
        self._lastlosses: Dict[str, float] = {
            lossname: 0.0 for lossname in conf.training.val_loss
        }

        for lossname, lossconf in conf.training.val_loss.items():
            assert lossname != "parts"
            params = lossconf if lossconf is not None else {}
            if not hasattr(torch.nn, lossname):
                loss = importlib.import_module(
                    f"fgsim.models.loss.{lossname}"
                ).LossGen(**params)
            else:
                loss = getattr(torch.nn, lossname)().to(device)
            self.parts[lossname] = loss
            setattr(self, lossname, loss)

    def __call__(self, holder, batch: BatchType) -> None:
        # During the validation, this function is called once per batch.
        # All losses are save in a dict for later evaluation log_lossses
        with torch.no_grad():
            for lossname, loss in self.parts.items():
                self._lastlosses[lossname] += float(loss(holder, batch))

    def log_losses(self, state) -> None:
        for lossname, loss in self._lastlosses.items():
            # Update the state
            state.val_losses[lossname].append(loss)
            # Log the validation loss
            self.train_logger.log_loss(f"loss.{self.name}.{lossname}", loss)
            # Reset to 0
            self._lastlosses[lossname] = 0

    def __getitem__(self, lossname: str) -> Callable:
        return self.parts[lossname]
