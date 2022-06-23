"""Dynamically import the losses"""
import importlib
import math
from typing import Dict, Optional, Protocol, Union

import torch
from omegaconf import DictConfig, OmegaConf

from fgsim.config import conf, device
from fgsim.io.sel_seq import Batch
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.utils.check_for_nans import contains_nans


class LossFunction(Protocol):
    def __call__(self, holder, batch: Batch) -> Union[float, Dict[str, float]]:
        ...


class SubNetworkLoss:
    """Holds all losses for a single subnetwork.
    Calling this class should return a single (1D) loss for the gradient step"""

    def __init__(
        self, subnetworkname: str, pconf: DictConfig, train_log: TrainLog
    ) -> None:
        self.name = subnetworkname
        self.pconf = pconf
        self.train_log = train_log
        self.parts: Dict[str, LossFunction] = {}

        for lossname, lossconf in pconf.items():
            assert lossname != "parts"
            params = lossconf if lossconf is not None else {}
            if isinstance(params, DictConfig):
                params = OmegaConf.to_container(params)
            if not hasattr(torch.nn, lossname):
                loss_module = importlib.import_module(
                    f"fgsim.models.loss.{lossname}"
                )
                loss = loss_module.LossGen(**params)
            else:
                loss = getattr(torch.nn, lossname)().to(device)
            self.parts[lossname] = loss
            setattr(self, lossname, loss)

    def __getitem__(self, lossname: str) -> LossFunction:
        return self.parts[lossname]

    def __call__(self, holder, batch: Batch):
        # In this dict comprehension iterates the losses for the current module
        # loss(holder, batch) returns either a float or
        # a dict with the sublosses subloss -> float
        # in the call loss(holder, batch), the .backward() call takes place
        lossesdict = {
            lossname: loss(holder, batch) for lossname, loss in self.parts.items()
        }
        losses_history = holder.history["losses"]
        # write the loss to the history so it can be logged later
        for lossname, loss in lossesdict.items():
            if isinstance(loss, float):
                self.log_loss(losses_history, loss, lossname)
            else:
                # For the case the loss return Dict[str,float]
                assert isinstance(loss, dict)
                # loss the sublosses
                for sublossname, sublossvalue in loss.items():
                    self.log_loss(
                        losses_history, sublossvalue, lossname, sublossname
                    )
                sumloss = sum(loss.values())
                self.log_loss(losses_history, sumloss, lossname, "sum")

        # Some of the losses are in dicts, other are not so we need to upack them
        summedloss = sum(
            [
                (lambda x: sum(list(x.values())) if isinstance(e, dict) else x)(e)
                for e in lossesdict.values()
            ]
        )
        assert not contains_nans(summedloss)[0]
        return summedloss

    def log_loss(
        self,
        losses_history: Dict,
        value: float,
        lossname: str,
        sublossname: Optional[str] = None,
    ):
        lossstr = (
            f"train.{self.name}.{lossname}"
            if sublossname is None
            else f"train.{self.name}.{lossname}.{sublossname}"
        )
        # Check for invalid values
        if math.isnan(value) or math.isinf(value):
            logger.error(f"{lossstr} returned invalid value: {value}")
            raise RuntimeError
        # Log to comet.ml
        self.train_log.log_loss(lossstr, value)
        # Make sure the fields in the state are available
        if sublossname is None:
            if lossname not in losses_history[self.name]:
                losses_history[self.name][lossname] = []
        else:
            if lossname not in losses_history[self.name]:
                losses_history[self.name][lossname] = {}
            if sublossname not in losses_history[self.name][lossname]:
                losses_history[self.name][lossname][sublossname] = []
        # Write values to the state
        if sublossname is None:
            losses_history[self.name][lossname].append(value)
        else:
            losses_history[self.name][lossname][sublossname].append(value)


class LossesCol:
    """Holds all losses for all subnetworks as attributes or as a dict."""

    def __init__(self, train_log: TrainLog) -> None:
        self.parts: Dict[str, SubNetworkLoss] = {}

        for subnetworkname, subnetworkconf in conf.models.items():
            snl = SubNetworkLoss(subnetworkname, subnetworkconf.losses, train_log)
            self.parts[subnetworkname] = snl
            setattr(self, subnetworkname, snl)

    def __getitem__(self, subnetworkname: str) -> SubNetworkLoss:
        return self.parts[subnetworkname]
