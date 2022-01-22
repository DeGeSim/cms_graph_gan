"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import awkward as ak
import numpy as np
import queueflow as qf
import torch
import uproot
import yaml
from torch.multiprocessing import Queue, Value

from fgsim.config import conf
from fgsim.geo.geo_lup import geo_lup

# Load files
ds_path = Path(conf.path.dataset)
assert ds_path.is_dir()
files = sorted(ds_path.glob(conf.path.dataset_glob))
if len(files) < 1:
    raise RuntimeError("No datasets found")

ChunkType = List[Tuple[Path, int, int]]

# load lengths
if not os.path.isfile(conf.path.ds_lenghts):
    len_dict = {}
    for fn in files:
        with uproot.open(fn) as rfile:
            len_dict[str(fn)] = rfile[conf.loader.rootprefix].num_entries
    ds_processed = Path(conf.path.dataset_processed)
    if not ds_processed.is_dir():
        ds_processed.mkdir()
    with open(conf.path.ds_lenghts, "w") as f:
        yaml.dump(len_dict, f, Dumper=yaml.SafeDumper)
else:
    with open(conf.path.ds_lenghts, "r") as f:
        len_dict = yaml.load(f, Loader=yaml.SafeLoader)


@dataclass
class Event:
    pc: torch.Tensor
    hlvs: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to(self, *args, **kwargs):
        self.pc = self.pc.to(*args, **kwargs)
        for key, val in self.hlvs.items():
            self.hlvs[key] = val.to(*args, **kwargs)
        return self

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

        for irow, key in enumerate(conf.loader.cell_prop_keys, start=1):
            vec = self.pc[:, irow]
            vec_ew = vec * e_weight
            mean = torch.mean(vec_ew)
            std = torch.std(vec_ew)
            self.hlvs[key + "_mean_ew"] = mean
            self.hlvs[key + "_std_ew"] = std
            # self.hlvs[key + "_mom3_ew"] = stand_mom(vec, mean, std, 3)
            # self.hlvs[key + "_mom4_ew"] = stand_mom(vec, mean, std, 4)
            for var, v in self.hlvs.items():
                if not var.startswith(key):
                    continue
                if torch.isnan(v):
                    raise ValueError("NaN Computed for hlv")


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


# Sharded switch for the postprocessing
postprocess_switch = Value("i", 0)


# Collect the steps
def process_seq():
    return (
        qf.ProcessStep(read_chunk, 2, name="read_chunk"),
        qf.PoolStep(
            transform,
            nworkers=conf.loader.num_workers_transform,
            name="transform",
        ),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(aggregate_to_batch, 1, name="batch"),
        # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    )


# Methods used in the Sequence

# reading from the filesystem
def read_chunk(chunks: ChunkType) -> ak.highlevel.Array:
    chunks_list = []
    for chunk in chunks:
        file_path, start, end = chunk
        with uproot.open(file_path) as rfile:
            roottree = rfile[conf.loader.rootprefix]
            chunks_list.append(
                roottree.arrays(
                    list(conf.loader.braches.values()),
                    entry_start=start,
                    entry_stop=end,
                    library="ak",
                )
            )

    # split up the events and pass them as a dict
    output = ak.concatenate(chunks_list)
    # remove the double gen energy
    return output


def aggregate_to_batch(list_of_graphs: List[Event]) -> Batch:
    batch = Batch.from_event_list(*list_of_graphs)
    return batch


def magic_do_nothing(batch: Batch) -> Batch:
    return batch


def transform(hitlist: ak.highlevel.Record) -> Event:
    pc = hitlist_to_pc(hitlist)
    event = Event(pc)
    if postprocess_switch.value:
        event.compute_hlvs()
    return event


def hitlist_to_pc(event: ak.highlevel.Record) -> torch.Tensor:
    key_id = conf.loader.braches.id
    key_hit_energy = conf.loader.braches.hit_energy

    # Sum up the sim  hits
    id_to_energy_dict: Dict[int, float] = {}

    for hit_energy, detid in zip(event[key_hit_energy], event[key_id]):
        # TODO fix the detids
        if detid not in geo_lup.index:
            continue
        if detid in id_to_energy_dict:
            id_to_energy_dict[detid] += hit_energy
        else:
            id_to_energy_dict[detid] = hit_energy

    detids = np.array(list(id_to_energy_dict.keys()), dtype=np.uint)

    # Filter out the rows/detids that are not in the event
    geo_lup_filtered = geo_lup.reindex(index=detids)

    # compute static features
    hit_energies = torch.tensor(
        list(id_to_energy_dict.values()), dtype=torch.float32
    )
    xyzpos = torch.tensor(
        geo_lup_filtered[conf.loader.cell_prop_keys].values, dtype=torch.float32
    )

    pc = torch.hstack((hit_energies.view(-1, 1), xyzpos))

    pc = pc.float()
    if conf.loader.n_points < pc.shape[0]:
        raise RuntimeError(
            "Event hast more points then the padding: "
            f"{conf.loader.n_points} < {pc.shape[0]}"
        )
    pc = torch.nn.functional.pad(
        pc,
        (0, 0, 0, conf.loader.n_points - pc.shape[0]),
        mode="constant",
        value=0,
    )
    if torch.any(pc[:, 0] < 0):
        raise Exception
    assert pc.shape[0] == conf.loader.n_points
    assert pc.shape[1] == conf.loader.n_features
    return pc


def stand_mom(
    vec: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, order: int
) -> torch.Tensor:
    numerator = torch.mean(torch.pow(vec - mean, order))
    denominator = torch.pow(std, order / 2.0)
    if numerator == denominator:
        return torch.tensor(1).float()
    if denominator == 0:
        raise ValueError("Would devide by 0.")
    return numerator / denominator
