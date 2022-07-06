from pathlib import Path
from typing import Dict, List, Tuple

import awkward as ak
import numpy as np
import queueflow as qf
import torch
import uproot
from torch.multiprocessing import Queue, Value

from fgsim.config import conf
from fgsim.geo.geo_lup import geo_lup

from .event import Batch, Event

# Sharded switch for the postprocessing
postprocess_switch = Value("i", 0)


ChunkType = List[Tuple[Path, int, int]]
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


# Collect the steps
def process_unscaled_seq():
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
    pointcloud = hitlist_to_pc(hitlist)
    event = Event(pointcloud)
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
        geo_lup_filtered[conf.loader.cell_prop_keys[1:]].values, dtype=torch.float32
    )

    pc = torch.hstack((hit_energies.view(-1, 1), xyzpos))

    pc = pc.float()
    if conf.loader.max_points < pc.shape[0]:
        raise RuntimeError(
            "Event hast more points then the padding: "
            f"{conf.loader.max_points} < {pc.shape[0]}"
        )
    pc = torch.nn.functional.pad(
        pc,
        (0, 0, 0, conf.loader.max_points - pc.shape[0]),
        mode="constant",
        value=0,
    )
    if torch.any(pc[:, 0] < 0):
        raise Exception
    assert pc.shape[0] == conf.loader.max_points
    assert pc.shape[1] == conf.loader.n_features
    return pc
