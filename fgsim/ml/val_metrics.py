"""Dynamically import the losses"""
import importlib
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from fgsim.config import conf
from fgsim.io.sel_loader import Batch
from fgsim.monitoring.metrics_aggr import MetricAggregator
from fgsim.monitoring.train_log import TrainLog


class ValidationMetrics:
    def __init__(self, train_log: TrainLog) -> None:
        self.train_log = train_log
        self.parts: Dict[str, Callable] = {}
        self._lastlosses: Dict[str, List[float]] = {}
        self.metric_aggr = MetricAggregator(train_log.history["val_metrics"])

        for metric_name, metric_conf in conf.training.val.metrics.items():
            assert metric_name != "parts"
            params = metric_conf if metric_conf is not None else DictConfig({})
            loss = import_metric(metric_name, params)

            self.parts[metric_name] = loss
            setattr(self, metric_name, loss)

    def __call__(
        self,
        gen_batch: Batch,
        sim_batch: Batch,
        d_gen: torch.Tensor,
        d_sim: torch.Tensor,
    ) -> None:
        # During the validation, this function is called once per batch.
        # All losses are save in a dict for later evaluation log_lossses
        kwargs = {
            "gen_batch": gen_batch,
            "sim_batch": sim_batch,
            "d_gen": d_gen,
            "d_sim": d_sim,
        }
        mval = {}
        with torch.no_grad():
            for metric_name, metric in self.parts.items():
                comp_metrics = metric(**kwargs)
                if isinstance(comp_metrics, dict):
                    # If the loss is processed for each hlv
                    # the return type is Dict[str,float]
                    for var, lossval in comp_metrics.items():
                        mval[var] = float(lossval)
                else:
                    mval[metric_name] = comp_metrics
        self.metric_aggr.append_dict(mval)

    def log_metrics(self) -> None:
        # Call metric_aggr to aggregate the collected matrics over the
        # validation batches. This will also update history["val_metrics"]
        up_metrics_d = self.metric_aggr.aggregate()

        for metric_name, metric_val in up_metrics_d.items():
            # Log the validation loss
            self.train_log.log_metric(f"val/{metric_name}", metric_val)

        # compute the stop_metric
        history = self.train_log.history
        val_metrics = history["val_metrics"]
        # collect all metrics for all validation runs in a 2d array
        loss_history = np.stack(
            [val_metrics[metric] for metric in conf.training.val.use_for_stopping]
        )
        # for a given metric and validation run,
        # count the fraction of times that the value of this metric
        # is smaller then the other runs
        ratio_better = np.apply_along_axis(
            lambda row: np.array([np.mean(row <= e) for e in row]), 1, loss_history
        ).mean(0)
        ratio_better = [float(val) for val in ratio_better]
        history["stop_crit"] = list(ratio_better)
        # overwrite the recorded ratio_better for each
        for ivalstep in range(len(ratio_better)):
            self.train_log.log_metric(
                name="val/ratio_better",
                value=ratio_better[ivalstep],
                step=ivalstep * conf.training.val.interval,
            )

    def __getitem__(self, lossname: str) -> Callable:
        return self.parts[lossname]


def import_metric(metric_name: str, params: DictConfig) -> Callable:
    try:
        metrics = importlib.import_module(f"fgsim.models.metrics")
        fct = getattr(metrics, metric_name)
        return fct
    except AttributeError:
        pass
    MetricClass: Optional = None
    for import_path in [
        f"torch.nn.{metric_name}",
        f"fgsim.models.metrics.{metric_name}",
    ]:
        try:
            model_module = importlib.import_module(import_path)
            # Check if it is a class
            if not isinstance(model_module, type):
                if not hasattr(model_module, "Metric"):
                    raise ModuleNotFoundError
                else:
                    MetricClass = model_module.Metric
            else:
                MetricClass = model_module

            break
        except ModuleNotFoundError:
            MetricClass = None

    if MetricClass is None:
        raise ImportError

    metric = MetricClass(**params)

    return metric
