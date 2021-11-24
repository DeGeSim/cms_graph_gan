"""Manages the networks for the holder class"""
import importlib
import os
from typing import Dict

import torch
from omegaconf import OmegaConf

from fgsim.config import conf, device
from fgsim.ml.loss import LossesCollector
from fgsim.utils.check_for_nans import contains_nans
from fgsim.utils.count_parameters import count_parameters
from fgsim.utils.logger import logger
from fgsim.utils.push_to_old import push_to_old


class SubNetwork(torch.nn.Module):
    def __init__(self, pconf: OmegaConf) -> None:
        super().__init__()
        self.pconf = pconf

        modelparams = self.pconf.param if self.pconf.param is not None else {}

        # Import the specified model
        self.model: torch.nn.Module = importlib.import_module(
            f"fgsim.models.{self.pconf.name}"
        ).ModelClass(**modelparams)
        count_parameters(self.model)

        optimparams = (
            self.pconf.optim.param if self.pconf.optim.param is not None else {}
        )
        self.optim = getattr(torch.optim, self.pconf.optim.name)(
            self.model.parameters(), **optimparams
        )

        self._lossf = LossesCollector(self.pconf.losses)

        # try to load a check point
        self.cppath = conf.path.checkpoint.format(self.pconf.name)
        self.checkpoint_loaded = False
        self.__load_checkpoint()

        # Move Model and optimizer parameters to the right device
        self.model = self.model.float().to(device)
        # Hack to move the optim parameters to the correct device
        # https://github.com/pytorch/pytorch/issues/8741
        self.optim.load_state_dict(self.optim.state_dict())

    def lossf(self, ytrue, ypred):
        loss = self._lossf(ytrue, ypred)
        assert not contains_nans(loss)[0]
        return loss

    def __load_checkpoint(self):
        if not os.path.isfile(self.cppath):
            logger.warning("Proceeding without loading checkpoint.")
            return

        checkpoint = torch.load(self.cppath, map_location=device)

        assert not contains_nans(checkpoint["model"])[0]
        assert not contains_nans(checkpoint["best_model"])[0]

        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optim"])
        self.best_model_state = checkpoint["best_model"]

        self.checkpoint_loaded = True

    def select_best_model(self):
        self.model.load_state_dict(self.best_model_state)
        self.model = self.model.float().to(device)

    def save_checkpoint(self):
        push_to_old(self.cppath, conf.path.checkpoint_old.format(self.pconf.name))

        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "best_model": self.best_model_state,
            },
            self.cppath,
        )

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)


class SubNetworkCollector(torch.nn.Module):
    """Collect all parts of the model in one to allow
    things like holder.model.to(device), float() ect."""

    def __init__(self, pconf: OmegaConf):
        self.pconf = pconf
        self.parts: Dict[str, torch.nn.Module] = {}
        for name, submodel in pconf.items():
            self.parts[name] = submodel.model
            setattr(self, name, submodel.model)

    def select_best_model(self):
        for submodel in self.parts.values():
            submodel.select_best_model()

    def save_checkpoint(self):
        for submodel in self.parts.values():
            submodel.save_checkpoint()

    def forward(self, X):
        for model in self.parts.values():
            X = model(X)
        return X
