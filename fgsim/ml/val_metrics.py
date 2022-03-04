"""Dynamically import the losses"""
import importlib
from typing import Callable, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from fgsim.config import conf
from fgsim.io.sel_seq import Batch
from fgsim.monitoring.train_log import TrainLog


class ValidationMetrics:
    def __init__(self, train_log: TrainLog) -> None:
        self.name = "val_loss"
        self.train_log = train_log
        self.parts: Dict[str, Callable] = {}
        self._lastlosses: Dict[str, float] = {}

        for metric_name, metric_conf in conf.validation.metrics.items():
            assert metric_name != "parts"
            params = metric_conf if metric_conf is not None else DictConfig({})
            loss = self.import_metric(metric_name, params)

            self.parts[metric_name] = loss
            setattr(self, metric_name, loss)

    def import_metric(self, metric_name: str, params: DictConfig) -> Callable:
        MetricClass: Optional = None
        for import_path in [
            f"torch.nn.{metric_name}",
            f"fgsim.models.metrics.{metric_name}",
        ]:
            try:
                model_module = importlib.import_module(import_path)
                # Check if it is a class
                if not isinstance(model_module, type):
                    if not hasattr(model_module, "LossGen"):
                        raise ModuleNotFoundError
                    else:
                        MetricClass = model_module.LossGen
                else:
                    MetricClass = model_module

                break
            except ModuleNotFoundError:
                MetricClass = None

        if MetricClass is None:
            raise ImportError

        metric = MetricClass(**params)

        return metric

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

    def log_losses(self, history) -> None:
        val_metrics = history["val_metrics"]
        for lossname, loss in self._lastlosses.items():
            # Update the state
            if lossname not in val_metrics:
                val_metrics[lossname] = []
            val_metrics[lossname].append(loss)
            # Reset to 0
            self._lastlosses[lossname] = 0

        # Log the validation loss
        if not conf.debug:
            with self.train_log.experiment.validate():
                for lossname, loss_history in val_metrics.items():
                    self.train_log.log_loss(
                        f"{self.name}.{lossname}", loss_history[-1]
                    )

        # compute the stop_metric
        loss_history = np.stack([val_metrics[metric] for metric in val_metrics])
        ratio_better = np.apply_along_axis(
            lambda row: np.array([np.mean(row <= e) for e in row]), 1, loss_history
        ).mean(0)
        ratio_better = [float(val) for val in ratio_better]
        history["stop_crit"] = list(ratio_better)
        if not conf.debug:
            for ivalstep in range(len(ratio_better)):
                # with self.train_log.experiment.validate():
                self.train_log.experiment.log_metric(
                    name="ratio_better",
                    value=ratio_better[ivalstep],
                    step=ivalstep * conf.validation.interval,
                )

    def __getitem__(self, lossname: str) -> Callable:
        return self.parts[lossname]
