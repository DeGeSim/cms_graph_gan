from collections import OrderedDict

import numpy as np


class MetricAggregator:
    def __init__(self) -> None:
        self.metric_collector = OrderedDict()

    def aggregate(self):
        aggr_dict = {k: np.mean(v) for k, v in self.metric_collector.items()}
        self.metric_collector = OrderedDict()
        return aggr_dict

    def append_dict(self, upd):
        # Make sure the fields in the state are available
        for metric_name in upd:
            if metric_name not in self.metric_collector:
                self.metric_collector[metric_name] = []
            # Write values to the state
            self.metric_collector[metric_name].append(upd[metric_name])


class GradHistAggregator:
    def __init__(self) -> None:
        self.metric_collector = OrderedDict()
        self.history = OrderedDict()

    def aggregate(self):
        aggr_dict = {k: np.mean(v) for k, v in self.metric_collector.items()}
        self.append_dict_(self.history, aggr_dict)
        self.metric_collector = OrderedDict()
        return self.history

    def append_dict(self, upd):
        self.append_dict_(self.metric_collector, upd)

    def append_dict_(self, target, upd):
        # Make sure the fields in the state are available
        for metric_name in upd:
            if metric_name not in target:
                target[metric_name] = []
            # Write values to the state
            target[metric_name].append(upd[metric_name])
