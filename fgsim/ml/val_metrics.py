"""Dynamically import the losses"""
import importlib
from typing import Callable, Dict

import numpy as np
import torch

from fgsim.config import conf
from fgsim.io.sel_seq import Batch
from fgsim.monitoring.train_log import TrainLog


class ValidationMetrics:
    def __init__(self, train_log: TrainLog) -> None:
        self.name = "val_loss"
        self.train_log = train_log
        self.parts: Dict[str, Callable] = {}
        self._lastlosses: Dict[str, float] = {}

        for lossname, lossconf in conf.validation.metrics.items():
            assert lossname != "parts"
            params = lossconf if lossconf is not None else {}
            filename = (
                params["function_file"] if "function_file" in params else lossname
            )
            if not hasattr(torch.nn, lossname):
                loss_class = importlib.import_module(
                    f"fgsim.models.loss.{filename}"
                ).LossGen
            else:
                loss_class = getattr(torch.nn, lossname)

            loss = loss_class(**params)
            self.parts[lossname] = loss
            setattr(self, lossname, loss)

    def __call__(self, holder, batch: Batch) -> None:
        # During the validation, this function is called once per batch.
        # All losses are save in a dict for later evaluation log_lossses
        with torch.no_grad():
            for lossname, loss in self.parts.items():
                if hasattr(loss, "foreach_hlv") and loss.foreach_hlv:
                    # If the loss is processed for each hlv
                    # the return type is Dict[str,float]
                    for var, lossval in loss(holder, batch).items():
                        lstr = f"{lossname}_{var}"
                        if lstr not in self._lastlosses:
                            self._lastlosses[lstr] = 0
                        if lossval in [float("nan"), float("inf"), float("-inf")]:
                            # raise ValueError(
                            #     f"Loss {lossname} evaluates to NaN for variable"
                            #     f" {var}: {lossval}."
                            # )
                            pass
                        else:
                            self._lastlosses[lstr] += float(lossval)
                else:
                    if lossname not in self._lastlosses:
                        self._lastlosses[lossname] = 0
                    self._lastlosses[lossname] += float(loss(holder, batch))

    def log_losses(self, state) -> None:
        for lossname, loss in self._lastlosses.items():
            # Update the state
            if lossname not in state.val_metrics:
                state.val_metrics[lossname] = []
            state.val_metrics[lossname].append(loss)
            # Reset to 0
            self._lastlosses[lossname] = 0

        # Log the validation loss
        if not conf.debug:
            with self.train_log.experiment.validate():
                for lossname, loss_history in state.val_metrics.items():
                    self.train_log.log_loss(
                        f"{self.name}.{lossname}", loss_history[-1]
                    )

        # compute the stop_metric
        loss_history = np.stack(
            [state.val_metrics[metric] for metric in state.val_metrics]
        )
        ratio_better = np.apply_along_axis(
            lambda row: np.array([np.mean(row <= e) for e in row]), 1, loss_history
        ).mean(0)
        ratio_better = [float(val) for val in ratio_better]
        state.stop_crit = list(ratio_better)
        if not conf.debug:
            with self.train_log.experiment.validate():
                self.train_log.experiment.log_curve(
                    "ratio_better",
                    x=np.arange(len(ratio_better)),
                    y=ratio_better,
                    overwrite=True,
                    step=self.train_log.state["grad_step"],
                )

    def __getitem__(self, lossname: str) -> Callable:
        return self.parts[lossname]
