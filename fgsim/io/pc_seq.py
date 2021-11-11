"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""
import os
from math import prod
from pathlib import Path
from typing import Dict, List, Tuple

import awkward as ak
import numpy as np
import queueflow as qf
import torch
import uproot
import yaml
from torch.multiprocessing import Queue

from fgsim.config import conf
from fgsim.geo.geo_lup import geo_lup

# Load files
ds_path = Path(conf.path.dataset)
assert ds_path.is_dir()
files = sorted(ds_path.glob(conf.path.dataset_glob))
if len(files) < 1:
    raise RuntimeError("No hdf5 datasets found")

ChunkType = List[Tuple[Path, int, int]]

# load lengths
if not os.path.isfile(conf.path.ds_lenghts):
    len_dict = {}
    for fn in files:
        with uproot.open(fn) as rfile:
            len_dict[str(fn)] = rfile[conf.loader.rootprefix].num_entries
    with open(conf.path.ds_lenghts, "w") as f:
        yaml.dump(len_dict, f, Dumper=yaml.SafeDumper)
else:
    with open(conf.path.ds_lenghts, "r") as f:
        len_dict = yaml.load(f, Loader=yaml.SafeLoader)

BatchType = torch.Tensor


npoints = prod(conf.models.gen.param.degrees)


# Collect the steps
def process_seq():
    return (
        qf.ProcessStep(read_chunk, 2, name="read_chunk"),
        qf.PoolStep(
            hitlist_to_pc,
            nworkers=conf.loader.num_workers_transform,
            name="transform",
        ),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(batch, 1, name="batch"),
        # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    )


# reading from the filesystem
def read_chunk(chunks: ChunkType) -> ak.highlevel.Array:
    chunks_list = []
    for chunk in chunks:
        file_path, start, end = chunk
        with uproot.open(file_path) as rfile:
            roottree = rfile[conf.loader.rootprefix]
            chunks_list.append(
                roottree.arrays(
                    conf.loader.keylist,
                    entry_start=start,
                    entry_stop=end,
                    library="ak",
                )
            )

    # split up the events and pass them as a dict
    output = ak.concatenate(chunks_list)
    # remove the double gen energy
    return output


def batch(list_of_graphs: List[torch.Tensor]) -> torch.Tensor:
    batch = torch.stack(list_of_graphs)
    return batch


def magic_do_nothing(batch: torch.Tensor) -> torch.Tensor:
    return batch


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
    padded_pc = torch.nn.functional.pad(
        pc, (0, 0, 0, npoints - pc.shape[0]), mode="constant", value=0
    )
    return padded_pc
