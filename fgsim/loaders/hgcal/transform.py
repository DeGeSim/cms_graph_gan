from functools import partial
from typing import Dict

import awkward as ak
import numpy as np
import torch
from torch_geometric.data import Data

from fgsim.config import conf
from fgsim.geo.geo_lup import geo_lup
from fgsim.monitoring.logger import logger


def transform_hitlist(
    event: ak.highlevel.Record, construct_edge_index: bool
) -> Data:
    key_id = conf.loader.braches.id
    key_hit_energy = conf.loader.braches.hit_energy

    # Multiple Hits can be in the same cell (simhits)
    # -> Sum up the sim hits
    id_to_energy_dict: Dict[int, float] = {}

    for hit_energy, detid in zip(event[key_hit_energy], event[key_id]):
        # TODO fix the detids
        if detid not in geo_lup.index:
            continue
        if detid in id_to_energy_dict:
            id_to_energy_dict[detid] += hit_energy
        else:
            id_to_energy_dict[detid] = hit_energy

    # ## Filtering of the detids
    # ### No filter
    # detids = list(id_to_energy_dict.keys())
    # ### n highest energies
    from heapq import nlargest

    detids = nlargest(
        conf.loader.n_points, id_to_energy_dict, key=id_to_energy_dict.get
    )
    # ### energy cut
    # # cut the hits 81% energy 56% of the hits @ 57 GeV
    # energyfilter = hit_energies > 0.0048
    # num_nodes = sum(energyfilter)
    # hit_energies = hit_energies[energyfilter]
    # detids = detids[energyfilter]

    # Filter out the rows/detids that are not in the event
    # also sorts the dataframe by the detids
    geo_lup_filtered = geo_lup.reindex(index=np.array(detids, dtype=np.uint))

    # check for NaNs of the detid is not present in the geolut
    check_columns = set(geo_lup_filtered.columns) & set(conf.loader.cell_prop_keys)
    props_df = geo_lup_filtered[list(check_columns)]
    # problemid= df[df.isnull().any(axis =1)].index[0] # 2160231891
    invalids = props_df[props_df.isnull().any(axis=1)].index
    if len(invalids) != 0:
        logger.error(f"No match in geo lut for detids {invalids}.")
        raise ValueError

    # ## Feature Matrix Construction
    hit_energies = torch.tensor(
        [id_to_energy_dict[e] for e in detids], dtype=torch.float32
    )
    xyzpos = torch.tensor(
        geo_lup_filtered[conf.loader.cell_prop_keys[1:]].values, dtype=torch.float32
    )

    pc = torch.hstack((hit_energies.view(-1, 1), xyzpos)).float()

    # ## Edge Index construction
    if construct_edge_index:
        neighbor_keys = [
            "next",
            "previous",
            "n0",
            "n1",
            "n2",
            "n3",
            "n4",
            "n5",
            "n6",
            "n7",
            "n8",
            "n9",
            "n10",
            "n11",
        ]
        neighbor_df = geo_lup_filtered[neighbor_keys]

        # number detids in events
        detid_lut = {det_id: i for i, det_id in enumerate(detids)}

        # ## construct adj mtx from detids
        # Select the detids from the current event

        node_neighbors_mtx_list = [
            np.vstack((np.repeat(detid, len(neighbor_keys)), neighbors_list.values))
            for detid, neighbors_list in neighbor_df.iterrows()
        ]
        edge_index_detid = np.hstack(node_neighbors_mtx_list)

        # Filter the out the zero entries:
        edge_index_detid = edge_index_detid[:, edge_index_detid[1] != 0]
        # Filter neigbhors not within the array
        eventid_set = set(detids)
        target_detid_in_event_mask = np.array(
            [(lambda x: x in eventid_set)(x) for x in edge_index_detid[1]]
        )
        edge_index_detid = edge_index_detid[:, target_detid_in_event_mask]

        # shift from detids to nodenumbers
        if edge_index_detid.size != 0:
            edge_index = np.vectorize(detid_lut.get)(edge_index_detid)
        else:
            edge_index = edge_index_detid

        # ## Build the graph
        graph = Data(
            x=pc,
            y=torch.tensor(event[conf.loader.braches.energy][0], dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
        )
    else:
        graph = Data(
            x=pc,
            y=torch.tensor(event[conf.loader.braches.energy][0], dtype=torch.float),
        )

    return graph


hitlist_to_pc = partial(transform_hitlist, construct_edge_index=False)
hitlist_to_graph = partial(transform_hitlist, construct_edge_index=True)
