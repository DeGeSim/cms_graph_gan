"""Manages the networks for the holder class"""
import importlib
from typing import Dict

import torch
from omegaconf.dictconfig import DictConfig


class SubNetworkCollector(torch.nn.Module):
    """Collect all parts of the model in one to allow
    things like holder.model.to(device), float() ect."""

    def __init__(self, pconf: DictConfig) -> None:
        super().__init__()
        self.pconf = pconf
        self.parts: Dict[str, torch.nn.Module] = {}

        for name, submodelconf in pconf.items():
            modelparams = (
                submodelconf.param if submodelconf.param is not None else {}
            )
            # Import the python file containing the models with dynamically
            submodel: torch.nn.Module = importlib.import_module(
                f"fgsim.models.subnetworks.{submodelconf.name}"
            ).ModelClass(**modelparams)

            self.parts[name] = submodel
            setattr(self, name, submodel)

    def forward(self, X):
        for model in self.parts.values():
            X = model(X)
        return X

    def get_par_dict(self) -> Dict:
        return {
            name: submodel.parameters() for name, submodel in self.parts.items()
        }

    def __getitem__(self, subnetworkname: str) -> torch.nn.Module:
        return self.parts[subnetworkname]
