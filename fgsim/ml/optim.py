"""Dynamically manages the optimizers"""
import importlib
from typing import Dict

import torch
from omegaconf import OmegaConf


class OptimCol:
    """Collect all optimizers for the different parts
    of the model to do eg. holder.model.zero_grad()."""

    def __init__(self, pconf: OmegaConf):

        self.pconf = pconf
        self.parts: Dict[str, torch.optim] = {}

        for name, submodelconf in pconf.items():
            assert name != "parts"
            params = submodelconf.optim if submodelconf.optim is not None else {}
            # Import the python file containing the models with dynamically
            optim: torch.optim = importlib.import_module(
                f"fgsim.models.{self.pconf.name}"
            ).ModelClass(**params)

            self.parts[name] = optim
            setattr(self, name, optim)

    def load_state_dict(self, state_dict):
        assert set(state_dict.keys()) == set(self.parts.keys())
        for name in state_dict.keys():
            self.parts[name].load_state_dict(state_dict[name])

    def state_dict(self):
        return {name: optim.state_dict() for name, optim in self.parts.items()}

    def zero_grad(self):
        for optim in self.parts.values():
            optim.zero_grad()

    def step(self):
        for optim in self.parts.values():
            optim.step()
