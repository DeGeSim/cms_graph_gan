from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from fgsim.config import conf
from fgsim.monitoring.logger import logger


@dataclass
class Event:
    pc: torch.Tensor
    hlvs: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, *args, **kwargs):
        self.pc = self.pc.to(*args, **kwargs)
        for key, val in self.hlvs.items():
            self.hlvs[key] = val.to(*args, **kwargs)
        return self

    def cpu(self):
        return self.to(torch.device("cpu"))

    def clone(self, *args, **kwargs):
        # This needs to return a new object to align the python and pytorch ref counts
        # Overwriting the attributes leads to memory leak with this
        # L = [event0,event1,event2,event3]
        # for e in L:
        #     e.to(gpu_device)

        pc = self.pc.clone(*args, **kwargs)
        hlvs = {key: val.clone(*args, **kwargs) for key, val in self.hlvs.items()}
        return type(self)(pc, hlvs)

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
        self, pc: torch.Tensor, hlvs: Optional[Dict[str, torch.Tensor]] = None
    ):
        if hlvs is None:
            hlvs = {}
        super().__init__(pc, hlvs)

    @classmethod
    def from_event_list(cls, *events: Event):
        pc = torch.stack([event.pc for event in events])
        hlvs = {
            key: torch.stack([event.hlvs[key] for event in events])
            for key in events[0].hlvs
        }
        return cls(pc=pc, hlvs=hlvs)

    def split(self) -> List[Event]:
        """Split batch into events."""
        outL = []
        for ievent in range(self.pc.shape[0]):
            e_pc = self.pc[ievent]
            e_hlvs = {key: self.hlvs[key][ievent] for key in self.hlvs}
            outL.append(Event(e_pc, e_hlvs))
        return outL

    def compute_hlvs(self) -> None:
        event_list = [x for x in self.split()]
        for event in event_list:
            event.compute_hlvs()
        self.hlvs = Batch.from_event_list(*event_list).hlvs
        return

    def to(self, *args, **kwargs):
        self.repad(conf.models.gen.output_points)
        return super().to(self, *args, **kwargs)

    def repad(self, new_size: int) -> None:
        if new_size < conf.loader.max_points:
            raise ValueError
        self.pc = torch.nn.functional.pad(
            self.pc,
            (0, 0, 0, new_size - conf.loader.max_points, 0, 0),
            mode="constant",
            value=0,
        )
