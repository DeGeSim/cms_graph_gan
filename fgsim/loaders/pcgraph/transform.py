from heapq import nlargest
from typing import Dict

import awkward as ak
import numpy as np
import torch

from fgsim.config import conf
from fgsim.geo.geo_lup import geo_lup


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

    # get detids with the the n highest energies
    detids_selected = nlargest(
        conf.loader.max_points, id_to_energy_dict, key=id_to_energy_dict.get
    )

    # Filter out the rows/detids that are not in the event
    geo_lup_filtered = geo_lup.reindex(
        index=np.array(detids_selected, dtype=np.uint)
    )

    # compute static features
    hit_energies = torch.tensor(
        [id_to_energy_dict[e] for e in detids_selected], dtype=torch.float32
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
    # Negative energies?
    if torch.any(pc[:, 0] < 0):
        raise Exception
    return pc
