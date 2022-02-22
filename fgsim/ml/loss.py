"""Dynamically import the losses"""
import importlib
from typing import Dict, Protocol, Union

import torch
from omegaconf.dictconfig import DictConfig

from fgsim.config import conf, device
from fgsim.io.sel_seq import Batch
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
            if not hasattr(torch.nn, lossname):
                loss = importlib.import_module(
                    f"fgsim.models.loss.{lossname}"
                ).LossGen(**params)
            else:
                loss = getattr(torch.nn, lossname)().to(device)
            self.parts[lossname] = loss
            setattr(self, lossname, loss)

    def __call__(self, holder, batch: Batch):
        lossesdict = {
            lossname: loss(holder, batch) for lossname, loss in self.parts.items()
        }
        # write the loss to state so it can be logged later
        for lossname, loss in lossesdict.items():
            if isinstance(loss, float):
                self.train_log.log_loss(f"loss.{self.name}.{lossname}", loss)
                holder.state.losses[self.name][lossname].append(loss)
            else:
                assert isinstance(loss, dict)
                # loss the sublosses
                for sublossname, subloss in loss.items():
                    self.train_log.log_loss(
                        f"loss.{self.name}.{lossname}.{sublossname}", subloss
                    )
                sumloss = sum(loss.values())
                self.train_log.log_loss(f"loss.{self.name}.{lossname}", sumloss)
                holder.state.losses[self.name][lossname].append(sumloss)

        # Some of the losses are in dicts, other are not so we need to upack them
        summedloss = sum(
            [
                (lambda x: sum(list(x.values())) if isinstance(e, dict) else x)(e)
                for e in lossesdict.values()
            ]
        )
        assert not contains_nans(summedloss)[0]
        return summedloss

    def __getitem__(self, lossname: str) -> LossFunction:
        return self.parts[lossname]


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
