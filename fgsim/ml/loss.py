"""Dynamically import the losses"""
import importlib
from typing import Dict, Optional, Protocol

import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

from fgsim.config import conf, device
from fgsim.monitoring.metrics_aggr import MetricAggregator
from fgsim.monitoring.train_log import TrainLog


class LossFunction(Protocol):
    def __call__(self, holder, batch: Batch) -> torch.Tensor:
        ...


class SubNetworkLoss:
    """Holds all losses for a single subnetwork.
    Calling this class should return a single (1D) loss for the gradient step
    Saves all losses in a MetricAggregator
    """

    def __init__(
        self, subnetworkname: str, pconf: DictConfig, train_log: TrainLog
    ) -> None:
        self.name = subnetworkname
        self.pconf = pconf
        self.train_log = train_log
        self.parts: Dict[str, LossFunction] = {}
        self.metric_aggr = MetricAggregator()
        # (train_log.history["losses"][self.name])

        for lossname, lossconf in pconf.items():
            assert lossname != "parts"
            params = lossconf if lossconf is not None else {}
            if isinstance(params, DictConfig):
                params = OmegaConf.to_container(params)
            del params["factor"]
            try:
                loss_module = importlib.import_module(
                    f"fgsim.models.loss.{lossname}"
                )
                loss = loss_module.LossGen(**params)
            except ImportError:
                loss = getattr(torch.nn, lossname)().to(device)
            self.parts[lossname] = loss
            setattr(self, lossname, loss)

    def __getitem__(self, lossname: str) -> LossFunction:
        return self.parts[lossname]

    def __call__(self, holder, **res):
        losses_dict: Dict[str, Optional[torch.Tensor]] = {}
        for lossname, loss in self.parts.items():
            loss_value = loss(holder=holder, **res)
            if loss_value.isnan():
                raise RuntimeError("Loss contains Nan")
            if loss_value is not None:
                losses_dict[lossname] = loss_value * self.pconf[lossname]["factor"]

        partloss: torch.Tensor = sum(losses_dict.values())
        if conf.models[self.name].retain_graph_on_backprop:
            partloss.backward(retain_graph=True)
        else:
            partloss.backward()
        self.metric_aggr.append_dict(
            {k: v.detach().cpu().numpy() for k, v in losses_dict.items()}
        )


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

    def __iter__(self):
        return iter(self.parts.values())
