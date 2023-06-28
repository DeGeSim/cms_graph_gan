"""Dynamically import the losses"""
import importlib

# from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from fgsim.config import conf
from fgsim.monitoring import MetricAggregator, TrainLog, logger


class ValidationMetrics:
    """
    Calling this object will save the output of the validation batches to
    self.metric_aggr . In the log_metrics they are aggregated and logged
    and a score is calculated.
    """

    def __init__(self, train_log: TrainLog, history) -> None:
        self.train_log = train_log
        self.parts: Dict[str, Callable] = {}
        self._lastlosses: Dict[str, List[float]] = {}
        self.metric_aggr = MetricAggregator()
        self.history = history

        metrics = (
            conf.training.val.debug_metrics
            if conf.debug
            else conf.training.val.metrics
        )
        for metric_name in metrics:
            assert metric_name != "parts"
            # params = metric_conf if metric_conf is not None else DictConfig({})
            params = DictConfig({})
            loss = import_metric(metric_name, params)

            self.parts[metric_name] = loss
            setattr(self, metric_name, loss)

    def __call__(self, **kwargs) -> None:
        # During the validation, this function is called once per batch.
        # All losses are save in a dict for later evaluation log_lossses
        mval = {}
        with torch.no_grad():
            for metric_name, metric in self.parts.items():
                # start = datetime.now()
                comp_metrics = metric(**kwargs)
                if isinstance(comp_metrics, dict):
                    # If the loss is processed for each hlv
                    # the return type is Dict[str,float]
                    for var, lossval in comp_metrics.items():
                        mval[f"{metric_name}_{var}"] = float(lossval)
                else:
                    mval[metric_name] = comp_metrics
                # print(
                #     f"Metric {metric_name} took"
                #     f" {(datetime.now()-start).seconds} sec"
                # )
        self.metric_aggr.append_dict(mval)

    def log_metrics(self, n_grad_steps_per_epoch, step) -> None:
        """
        The function takes the validation metrics and computes the fraction
        of times that the value of this metric is smaller then the other runs

        @param n_grad_steps_per_epoch The number of gradient steps per epoch.
        """
        # Call metric_aggr to aggregate the collected metrics over the
        # validation batches.
        up_metrics_d = self.metric_aggr.aggregate()

        # Log the validation loss
        self.train_log.log_metrics(up_metrics_d, prefix="val", step=step)

        logstr = "Validation:"
        for metric_name, metric_val in up_metrics_d.items():
            val_metric_hist = self.history["val"][metric_name]
            val_metric_hist.append(metric_val)

            logstr += f" {metric_name} {val_metric_hist[-1]:.2f} "
            if len(val_metric_hist) > 1:
                logstr += (
                    f"(Δ{(val_metric_hist[-1]/val_metric_hist[-2]-1)*100:+.0f}%)"
                )
        logger.warn(logstr)
        if conf.debug:
            return

        # compute the stop_metric
        val_metrics = self.history["val"]
        # check which of the metrics should be used for the early stopping
        # If a metric returns a dict, use all
        val_metrics_names = [
            k
            for k in up_metrics_d.keys()
            if any([k.startswith(mn) for mn in conf.training.val.use_for_stopping])
        ]
        # collect all metrics for all validation runs in a 2d array
        loss_history = np.stack(
            [val_metrics[metric] for metric in val_metrics_names]
        )
        # for a given metric and validation run,
        # count the fraction of times that the value of this metric
        # is smaller then the other runs
        score = np.apply_along_axis(
            lambda row: np.array([np.mean(row >= e) for e in row]), 1, loss_history
        ).mean(0)
        score = [float(val) for val in score]
        self.history["score"] = list(score)
        # overwrite the recorded score for each
        for ivalstep in range(len(score)):
            self.train_log.log_metrics(
                {"trend/score": score[ivalstep]},
                step=ivalstep * conf.training.val.interval,
                epoch=(ivalstep * conf.training.val.interval)
                // n_grad_steps_per_epoch,
            )

    def __getitem__(self, lossname: str) -> Callable:
        return self.parts[lossname]


def import_metric(metric_name: str, params: DictConfig) -> Callable:
    try:
        metrics = importlib.import_module("fgsim.models.metrics")
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
