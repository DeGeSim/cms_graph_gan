"""Manages the networks for the holder class"""
import importlib
from typing import Dict

import torch
from omegaconf import OmegaConf


class SubNetworkCollector(torch.nn.Module):
    """Collect all parts of the model in one to allow
    things like holder.model.to(device), float() ect."""

    def __init__(self, pconf: OmegaConf):
        self.pconf = pconf
        self.parts: Dict[str, torch.nn.Module] = {}

        for name, submodelconf in pconf.items():
            modelparams = (
                submodelconf.param if submodelconf.param is not None else {}
            )
            # Import the python file containing the models with dynamically
            submodel: torch.nn.Module = importlib.import_module(
                f"fgsim.models.{self.pconf.name}"
            ).ModelClass(**modelparams)

            self.parts[name] = submodel
            setattr(self, name, submodel)

    # def select_best_model(self):
    #     for submodel in self.parts.values():
    #         submodel.select_best_model()

    # def save_checkpoint(self):
    #     for submodel in self.parts.values():
    #         submodel.save_checkpoint()

    def forward(self, X):
        for model in self.parts.values():
            X = model(X)
        return X
