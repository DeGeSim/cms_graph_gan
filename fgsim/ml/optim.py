"""Dynamically manages the optimizers"""
from typing import Dict

import torch
from omegaconf.dictconfig import DictConfig

from fgsim.monitoring.metrics_aggr import MetricAggregator
from fgsim.monitoring.train_log import TrainLog


class OptimAndSchedulerCol:
    """Collect all optimizers for the different parts
    of the model to do eg. holder.model.zero_grad()."""

    def __init__(
        self, pconf: DictConfig, submodelpar_dict: Dict, train_log: TrainLog
    ):
        self.pconf = pconf
        self.train_log = train_log
        self.metric_aggr = MetricAggregator()
        self._optimizers: Dict[str, torch.optim.Optimizer] = {}
        self._schedulers: Dict[str, torch.optim.lr_scheduler._LRScheduler] = {}

        for name, submodelconf in pconf.items():
            assert name != "parts"
            optimparams = (
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
                    submodelpar_dict[name], **optimparams
                )
            self._optimizers[name] = optim

            # Setup the scheduler
            if submodelconf.scheduler.name == "NullScheduler":
                continue
            schedulerparams = (
                submodelconf.scheduler.params
                if submodelconf.scheduler.params is not None
                else {}
            )
            if "max_lr_factor" in schedulerparams:
                schedulerparams["max_lr"] = (
                    schedulerparams["max_lr_factor"] * optimparams["lr"]
                )
                del schedulerparams["max_lr_factor"]
            scheduler = getattr(
                torch.optim.lr_scheduler, submodelconf.scheduler.name
            )(optimizer=optim, **schedulerparams)
            self._schedulers[name] = scheduler

    def load_state_dict(self, state_dict):
        optims_state = state_dict["optimizers"]
        schedulers_state = state_dict["schedulers"]

        assert set(optims_state.keys()) == set(self._optimizers.keys())
        for name in optims_state.keys():
            self._optimizers[name].load_state_dict(optims_state[name])

        assert set(schedulers_state.keys()) == set(self._schedulers.keys())
        for name in schedulers_state.keys():
            self._schedulers[name].load_state_dict(schedulers_state[name])

    def state_dict(self) -> dict:
        outd = {}
        outd["optimizers"] = {
            name: optim.state_dict() for name, optim in self._optimizers.items()
        }
        outd["schedulers"] = {
            name: scheduler.state_dict()
            for name, scheduler in self._schedulers.items()
        }
        return outd

    def zero_grad(self, *args, **kwargs):
        for optim in self._optimizers.values():
            optim.zero_grad(*args, **kwargs)

    def step(self, pname):
        # optimizer step
        self._optimizers[pname].step()

        if pname not in self._schedulers:
            return

        try:
            self._schedulers[pname].step()
        except ValueError:
            pass
        lrs = self._schedulers[pname].get_last_lr()
        assert len(lrs) == 1
        self.metric_aggr.append_dict({pname: lrs[0]})

    def __getitem__(self, subnetworkname: str) -> torch.optim.Optimizer:
        return self._optimizers[subnetworkname]

    def to(self, device) -> None:
        for optim in self._optimizers.values():
            if isinstance(optim, FakeOptimizer):
                continue
            for param in optim.state.values():
                # Not sure there are any global tensors in the state dict
                if isinstance(param, torch.Tensor):
                    param.data = param.data.to(device)
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to(device)
                elif isinstance(param, dict):
                    for subparam in param.values():
                        if isinstance(subparam, torch.Tensor):
                            subparam.data = subparam.data.to(device)
                            if subparam._grad is not None:
                                subparam._grad.data = subparam._grad.data.to(device)


class FakeOptimizer(torch.optim.Optimizer):
    def __init__(self, *args, **kwargs) -> None:
        pass

    def zero_grad(self, *args, **kwargs) -> None:
        pass

    def step(self, *args, **kwargs) -> None:
        pass

    def state_dict(self, *args, **kwargs):
        return {}

    def load_state_dict(self, *args, **kwargs) -> None:
        pass
