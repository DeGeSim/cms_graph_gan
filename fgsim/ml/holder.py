import importlib
import os
from typing import Dict

import torch
from omegaconf import OmegaConf

from fgsim.config import conf, device
from fgsim.utils.check_for_nans import contains_nans
from fgsim.utils.count_parameters import count_parameters
from fgsim.utils.logger import logger


def push_to_old(path_new, path_old):
    if os.path.isfile(path_new):
        if os.path.isfile(path_old):
            os.remove(path_old)
        os.rename(path_new, path_old)


class SubModelHolder:
    def __init__(self, pconf) -> None:
        self.pconf = pconf

        modelparams = self.pconf.param if self.pconf.param is not None else {}
        optimparams = (
            self.pconf.optim.param if self.pconf.optim.param is not None else {}
        )
        # Import the specified model
        self.model: torch.nn.Module = importlib.import_module(
            f"fgsim.models.{self.pconf.name}"
            # f"fgsim.models.{self.pconf.name}", "fgsim.models"
        ).ModelClass(**modelparams)
        count_parameters(self.model)

        self.optim = getattr(torch.optim, self.pconf.optim.name)(
            self.model.parameters(), **optimparams
        )

        self._lossf = getattr(torch.nn, self.pconf.loss.name)().to(device)

        # try to load a check point
        self.cppath = conf.path.checkpoint.format(self.pconf.name)
        self.checkpoint_loaded = False
        self.__load_checkpoint()

        # Move Model and optimizer parameters to the right device
        self.model = self.model.float().to(device)
        # Hack to move the optim parameters to the correct device
        # https://github.com/pytorch/pytorch/issues/8741
        self.optim.load_state_dict(self.optim.state_dict())

    def lossf(self, y, yhat):
        loss = self._lossf(y, yhat)
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
        self.push_to_old(
            self.cppath, conf.path.checkpoint_old.format(self.pconf.name)
        )

        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "best_model": self.best_model_state,
            },
            self.cppath,
        )


class SubModelCollector(torch.nn.Module):
    # Collect all parts of the model in one to allow
    # things like holder.model.to(device), float() ect.
    def __init__(self, subholderdict):
        super().__init__()
        self._parts: Dict[str, torch.nn.Module] = {}
        for name, submodel in subholderdict.items():
            self._parts[name] = submodel.model
            setattr(self, name, submodel.model)


class OptimCollector:
    # Collect all optimizers for the different parts
    # of the model to do eg. holder.model.zero_grad()
    def __init__(self, subholderdict):
        self._parts: Dict[str, torch.optim] = {}
        for name, submodel in subholderdict.items():
            self._parts[name] = submodel.optim
            setattr(self, name, submodel.optim)

    def zero_grad(self):
        for _, optim in self._parts.items():
            optim.zero_grad()

    def step(self):
        for _, optim in self._parts.items():
            optim.step()


class ModelHolder:
    """ "This class holds the model the loss function and the optim.
    It manages the checkpointing and holds a member 'state' that contains
    information about the current state of the training"""

    def __init__(self) -> None:
        self.state = OmegaConf.create(
            {
                "epoch": 0,
                "processed_events": 0,
                "ibatch": 0,
                "grad_step": 0,
                "val_losses": [],
            }
        )
        self.submodels: Dict[str, SubModelHolder] = {}
        for name, pconf in conf.models.items():
            self.submodels[name] = SubModelHolder(pconf)
            setattr(self, name, self.submodels[name].model)

        self.model = SubModelCollector(self.submodels)
        self.optim = OptimCollector(self.submodels)

        self.__load_checkpoint()
        checkpoint_loaded = next(iter(self.submodels.values())).checkpoint_loaded
        assert all(
            checkpoint_loaded == e.checkpoint_loaded
            for e in self.submodels.values()
        ), (
            "Some parts of the model have been loaded from the checkpoint, others"
            " not "
        )
        if checkpoint_loaded:
            logger.warning(
                "Loading model from checkpoint at"
                + f" epoch {self.state['epoch']}"
                + f" batch {self.state['ibatch']}"
                + f" grad_step {self.state['grad_step']}."
            )
        else:
            logger.warning("Starting new training from scratch.")

    def __load_checkpoint(self):
        if not os.path.isfile(conf.path.state):
            logger.warning("Proceeding without loading checkpoint.")
            return

        self.state = OmegaConf.load(conf.path.state)

    def select_best_model(self):
        for e in self.submodels.values():
            e.select_best_model()

    def save_checkpoint(self):
        self.push_to_old(conf.path.state, conf.path.state_old)
        OmegaConf.save(config=self.state, f=conf.path.state)
        for e in self.submodels.values():
            e.save_checkpoint()


model_holder = ModelHolder()
