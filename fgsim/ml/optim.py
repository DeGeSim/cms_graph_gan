"""Dynamically manages the optimizers"""
from typing import Dict

import torch
from omegaconf.dictconfig import DictConfig


class OptimCol:
    """Collect all optimizers for the different parts
    of the model to do eg. holder.model.zero_grad()."""

    def __init__(self, pconf: DictConfig, submodelpar_dict: Dict):

        self.pconf = pconf
        self.parts: Dict[str, torch.optim.Optimizer] = {}

        for name, submodelconf in pconf.items():
            assert name != "parts"
            params = (
                submodelconf.optim.params
                if submodelconf.optim.params is not None
                else {}
            )

            # Import the python file containing the models with dynamically
            optim: torch.optim = getattr(torch.optim, submodelconf.optim.name)(
                submodelpar_dict[name], **params
            )

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

    def __getitem__(self, subnetworkname: str) -> torch.optim.Optimizer:
        return self.parts[subnetworkname]
