"""Dynamically import the losses"""
import importlib
from datetime import datetime

# from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from fgsim.config import conf
from fgsim.monitoring import MetricAggregator, TrainLog, logger


class EvaluationMetrics:
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
                logger.debug(f"Running metric {metric_name}")
                start = datetime.now()
                comp_metrics = metric(**kwargs)
                delta = datetime.now() - start
                if delta.seconds > 10:
                    logger.info(
                        f"Metric {metric_name} took {delta.total_seconds()} sec"
                    )
                if isinstance(comp_metrics, dict):
                    # If the loss is processed for each hlv
                    # the return type is Dict[str,float]
                    for var, lossval in comp_metrics.items():
                        mval[f"{metric_name}/{var}"] = float(lossval)
                else:
                    mval[metric_name] = comp_metrics

        self.metric_aggr.append_dict(mval)

    def get_metrics(self) -> tuple[dict, list]:
        """
        The function takes the validation metrics and computes the fraction
        of times that the value of this metric is smaller then the other runs
        """
        # Call metric_aggr to aggregate the collected metrics over the
        # validation batches.
        up_metrics_d = self.__aggr_dists(self.metric_aggr.aggregate())

        for metric_name, metric_val in up_metrics_d.items():
            val_metric_hist = self.history["val"][metric_name]
            val_metric_hist.append(metric_val)

            logstr = f"Validation: {metric_name} {val_metric_hist[-1]:.2f} "
            if len(val_metric_hist) > 1:
                logstr += (
                    f"(Δ{(val_metric_hist[-1]/val_metric_hist[-2]-1)*100:+.0f}%)"
                )
            logger.info(logstr)
        if conf.debug:
            return dict(), list()

        score = self.__compute_score_per_val(up_metrics_d)
        self.history["score"] = score
        return up_metrics_d, score

    def __aggr_dists(self, md):
        for dname in ["cdf", "sw1", "histd"]:
            md[f"dmean_{dname}"] = float(
                np.nanmean([v for k, v in md.items() if k.endswith(dname)])
            )
        return md

    def __compute_score_per_val(self, up_metrics_d):
        # compute the stop_metric
        val_metrics = self.history["val"]
        # check which of the metrics should be used for the early stopping
        # If a metric returns a dict, use all
        val_metrics_names = [
            k
            for k in up_metrics_d.keys()
            if any([k.startswith(mn) for mn in conf.training.val.use_for_stopping])
        ]

        # for the following, all recordings need to have the same
        # lenght, so we count the most frequent one
        histlen = max([len(val_metrics[metric]) for metric in val_metrics_names])
        # collect all metrics for all validation runs in a 2d array
        loss_history = np.stack(
            [
                val_metrics[metric]
                for metric in val_metrics_names
                if len(val_metrics[metric]) == histlen
            ]
        )
        # for a given metric and validation run,
        # count the fraction of times that the value of this metric
        # is smaller then the other runs
        score = np.apply_along_axis(
            lambda row: np.array([np.mean(row >= e) for e in row]), 1, loss_history
        ).mean(0)
        score = [float(val) for val in score]
        return score

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
