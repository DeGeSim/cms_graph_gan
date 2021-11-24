"""Dynamically manages the optimizers"""
from typing import Dict

import torch


class OptimCol:
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
