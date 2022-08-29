"""Dynamically manages the optimizers"""
from typing import Dict

import torch
from omegaconf.dictconfig import DictConfig

from fgsim.utils.optimto import optimizer_to


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

            optim: torch.optim
            if submodelconf.optim.name == "FakeOptimizer":
                optim = FakeOptimizer()
            else:
                # Import the python file containing the models with dynamically
                optim = getattr(torch.optim, submodelconf.optim.name)(
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

    def to(self, device) -> None:
        for optim in self.parts.values():
            if not isinstance(optim, FakeOptimizer):
                optimizer_to(optim, device)


class FakeOptimizer(torch.optim.Optimizer):
    def __init__(self) -> None:
        pass

    def zero_grad(self) -> None:
        pass

    def step(self) -> None:
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, *args, **kwargs) -> None:
        pass
