"""This modules manages all objects that need to be available for the training:
Subnetworks, losses and optimizers. The Subnetworks and losses are dynamically imported,
depending on the config. Contains the code for checkpointing of model and optimzer status."""

import importlib
import os
from typing import Callable, Dict, List

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


class LossesCollector:
    """Holds all losses for a single subnetwork.
    Calling this class should return a single (1D) loss for the gradient step"""

    def __init__(self, pconf: OmegaConf) -> None:
        self.pconf = pconf
        self.losses: List[Callable] = [Loss(e) for e in pconf]

    def __call__(self, ytrue, ypred):
        return torch.sum(loss(ytrue, ypred) for loss in self.losses)


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
        ).ModelClass(**modelparams)
        count_parameters(self.model)

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


class SubModelCollector(torch.nn.Module):
    """Collect all parts of the model in one to allow
    things like holder.model.to(device), float() ect."""

    def __init__(self, subholderdict):
        super().__init__()
        self._parts: Dict[str, torch.nn.Module] = {}
        for name, submodel in subholderdict.items():
            self._parts[name] = submodel.model
            setattr(self, name, submodel.model)


class OptimCollector:
    """Collect all optimizers for the different parts
    of the model to do eg. holder.model.zero_grad()."""

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
        for submod in self.submodels.values():
            submod.select_best_model()

    def save_checkpoint(self):
        push_to_old(conf.path.state, conf.path.state_old)
        OmegaConf.save(config=self.state, f=conf.path.state)
        for submod in self.submodels.values():
            submod.save_checkpoint()


model_holder = ModelHolder()
