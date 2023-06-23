from collections import OrderedDict, deque

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
                self.metric_collector[metric_name] = deque()
            # Write values to the state
            self.metric_collector[metric_name].append(upd[metric_name])


class GradHistAggregator:
    def __init__(self) -> None:
        self.metric_collector = OrderedDict()
        self.history = OrderedDict()
        self.steps = deque()
        # self.max_memory = 10

    def aggregate(self, step):
        aggr_dict = {k: np.mean(v) for k, v in self.metric_collector.items()}
        self.append_dict_(self.history, aggr_dict)
        self.steps.append(step)
        # if len(list(self.history.values())[0]) > self.max_memory:
        #     self.compress_history()
        self.metric_collector = OrderedDict()
        return self.history

    def append_dict(self, upd):
        self.append_dict_(self.metric_collector, upd)

    def append_dict_(self, target, upd):
        # Make sure the fields in the state are available
        for metric_name in upd:
            if metric_name not in target:
                target[metric_name] = deque()
            # Write values to the state
            target[metric_name].append(upd[metric_name])

    # def compress_history(self):
    #     for k, v in self.history.items():
    #         self.history[k] = deque(
    #             list(np.array(v).reshape(2, -1)[0, -1].reshape(-1)),
    #  self.max_memory
    #         )
