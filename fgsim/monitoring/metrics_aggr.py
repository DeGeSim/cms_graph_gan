import numpy as np


class MetricAggregator:
    def __init__(self, history) -> None:
        self.history = history
        self.metric_collector = {}

    def aggregate(self):
        aggr_dict = {k: np.mean(v) for k, v in self.metric_collector.items()}
        for k, v in aggr_dict.items():
            self.history[k].append(v)
        self.metric_collector = {}
        return aggr_dict

    def append_dict(self, upd):
        # Make sure the fields in the state are available
        for metric_name in upd:
            if metric_name not in self.history:
                self.history[metric_name] = []
            if metric_name not in self.metric_collector:
                self.metric_collector[metric_name] = []
            # Write values to the state
            self.metric_collector[metric_name].append(upd[metric_name])
