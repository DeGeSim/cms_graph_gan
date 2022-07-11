from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Batch as GraphBatch
from torch_geometric.data import Data as Graph

from fgsim.config import conf
from fgsim.monitoring.logger import logger


@dataclass
class Event:
    graph: Graph
    hlvs: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def pc(self):
        return self.graph.x

    def to(self, *args, **kwargs):
        self.graph = self.graph.to(*args, **kwargs)
        for key, val in self.hlvs.items():
            self.hlvs[key] = val.to(*args, **kwargs)
        return self

    def cpu(self):
        return self.to(torch.device("cpu"))

    def clone(self, *args, **kwargs):
        graph = self.graph.clone(*args, **kwargs)
        hlvs = {key: val.clone(*args, **kwargs) for key, val in self.hlvs.items()}
        return type(self)(graph, hlvs)

    def compute_hlvs(self) -> None:
        self.hlvs["energy_sum"] = torch.sum(self.pc[:, 0])
        self.hlvs["energy_sum_std"] = torch.std(self.pc[:, 0])
        e_weight = self.pc[:, 0] / self.hlvs["energy_sum"]
        # if torch.isnan(torch.sum(e_weight)):
        #     logger.warning(f"energy: vec {min_mean_max(self.pc[:, 0])}")
        #     logger.warning(f"x: vec {min_mean_max(self.pc[:, 1])}")
        #     logger.warning(f"y: vec {min_mean_max(self.pc[:, 2])}")
        #     logger.warning(f"z: vec {min_mean_max(self.pc[:, 3])}")

        for irow, key in enumerate(conf.loader.cell_prop_keys):
            if key == "E":
                continue
            vec = self.pc[:, irow]

            vec_ew = vec * e_weight
            mean = torch.mean(vec_ew)
            std = torch.std(vec_ew)
            self.hlvs[key + "_mean_ew"] = mean
            self.hlvs[key + "_std_ew"] = std
            # self.hlvs[key + "_mom3_ew"] = self._stand_mom(vec, mean, std, 3)
            # self.hlvs[key + "_mom4_ew"] = self._stand_mom(vec, mean, std, 4)
            for var, v in self.hlvs.items():
                if not var.startswith(key):
                    continue
                if torch.isnan(v):
                    logger.error(f"NaN Computed for hlv {var}")

    def _stand_mom(
        self, vec: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, order: int
    ) -> torch.Tensor:
        numerator = torch.mean(torch.pow(vec - mean, order))
        denominator = torch.pow(std, order / 2.0)
        if numerator == denominator:
            return torch.tensor(1).float()
        if denominator == 0:
            raise ValueError("Would devide by 0.")
        return numerator / denominator


def min_mean_max(vec):
    return (
        f"min {torch.min(vec)} mean {torch.mean(vec)} max"
        f" {torch.max(vec)} nan%{sum(torch.isnan(vec))/sum(vec.shape)}"
    )


class Batch(Event):
    def __init__(
        self, graph: Graph, hlvs: Optional[Dict[str, torch.Tensor]] = None
    ):
        self.graph = graph
        if hlvs is None:
            self.hlvs = {}
        else:
            self.hlvs = hlvs

    @classmethod
    def from_event_list(cls, *events: Event):
        graph = GraphBatch.from_data_list([event.graph for event in events])
        hlvs = {
            key: torch.stack([event.hlvs[key] for event in events])
            for key in events[0].hlvs
        }
        return cls(graph=graph, hlvs=hlvs)

    @classmethod
    def from_pcs_list(cls, pcs: torch.Tensor, events: torch.Tensor):
        pcs_list = [pcs[events == ievent] for ievent in range(max(events) + 1)]
        event_list = [Event(Graph(x=pc)) for pc in pcs_list]

        return cls.from_event_list(*event_list)

    def split(self) -> List[Event]:
        """Split batch into events."""
        graphs_list = self.graph.to_data_list()
        outL = []
        for ievent in range(self.graph.num_graphs):
            e_graph = graphs_list[ievent]
            e_hlvs = {key: self.hlvs[key][ievent] for key in self.hlvs}
            outL.append(Event(e_graph, e_hlvs))
        return outL

    def compute_hlvs(self) -> None:
        event_list = [x for x in self.split()]
        for event in event_list:
            event.compute_hlvs()
        self.hlvs = Batch.from_event_list(*event_list).hlvs
        return
