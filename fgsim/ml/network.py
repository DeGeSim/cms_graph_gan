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
            submodel = import_nn(submodelconf.name, modelparams)

            self.parts[name] = submodel
            setattr(self, name, submodel)

    def forward(self, in_X):
        for model in self.parts.values():
            in_X = model(in_X)
        return in_X

    def get_par_dict(self) -> Dict:
        return {
            name: submodel.parameters() for name, submodel in self.parts.items()
        }

    def __getitem__(self, subnetworkname: str) -> torch.nn.Module:
        return self.parts[subnetworkname]


# Import the python file containing the models with dynamically
def import_nn(nn_name, modelparams) -> torch.nn.Module:
    try:
        model_module = importlib.import_module(
            f"fgsim.models.subnetworks.{nn_name}"
        )
        if not hasattr(model_module, "ModelClass"):
            raise ModuleNotFoundError
        submodel = model_module.ModelClass(**modelparams)
    except ModuleNotFoundError:
        # Submodule with a model.py with the ModelClass
        try:
            model_module = importlib.import_module(
                f"fgsim.models.subnetworks.{nn_name}.model"
            )
            if not hasattr(model_module, "ModelClass"):
                raise ModuleNotFoundError
            submodel = model_module.ModelClass(**modelparams)
        except ModuleNotFoundError:
            # Submodule with a {nn_name}.py with the ModelClass
            try:
                model_module = importlib.import_module(
                    f"fgsim.models.subnetworks.{nn_name}.{nn_name}"
                )
                if not hasattr(model_module, "ModelClass"):
                    raise ModuleNotFoundError
                submodel = model_module.ModelClass(**modelparams)
            except ModuleNotFoundError:
                raise ModuleNotFoundError
    return submodel
