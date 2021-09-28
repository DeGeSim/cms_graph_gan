"""
Conversion from list of hit to graph
"""

# import sys

from os.path import isfile, splitext
from typing import Dict

import awkward as ak
import numpy as np
import pandas as pd
import torch
import torch_geometric
import uproot

# import yappi
from torch_geometric.data import Data as GraphType

from fgsim.config import conf
from fgsim.utils.logger import logger

pickle_lup_path = splitext("conf.path.geo_lup")[0] + ".pd"
if not isfile(pickle_lup_path):
    with uproot.open(conf.path.geo_lup) as rf:
        geo_lup = rf["analyzer/tree;1"].arrays(library="ak")
    geo_lup = ak.to_pandas(geo_lup)
    geo_lup.set_index("globalid", inplace=True)
    geo_lup.to_pickle(pickle_lup_path)

geo_lup = pd.read_pickle(pickle_lup_path)


def event_to_graph(event: ak.highlevel.Record) -> GraphType:
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
    hit_energies = np.array(list(id_to_energy_dict.values()), dtype=np.float32)

    # assert np.all(hit_energies==[id_to_energy_dict[x] for x in detids])

    # cut the hits 81% energy 56% of the hits @ 57 GeV
    energyfilter = hit_energies > 0  # 0.0048
    num_nodes = sum(energyfilter)
    hit_energies = hit_energies[energyfilter]
    detids = detids[energyfilter]

    # _, counts = np.unique(ak.to_numpy(detids), return_counts=True)
    # assert all(counts==1)

    # Filter out the rows/detids that are not in the event
    geo_lup_filtered = geo_lup.reindex(index=detids)
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

    # compute static features

    static_feature_matrix = geo_lup_filtered[conf.loader.cell_prop_keys].values

    # check for NaNs of the detid is not present in the geolut
    props_df = geo_lup_filtered[conf.loader.cell_prop_keys]
    # problemid= df[df.isnull().any(axis =1)].index[0] # 2160231891
    invalids = props_df[props_df.isnull().any(axis=1)].index
    if len(invalids) != 0:
        logger.error(f"No match in geo lut for detids {invalids}.")
        raise ValueError
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

    # Collects the hlvs now:
    hlvs = {}
    hlvs["sum_energy"] = sum(hit_energies)
    detid_isolated = set(detids) - set(np.unique(edge_index))
    hlvs["num_isolated"] = len(detid_isolated)
    hlvs["isolated_energy"] = sum(
        [hit_energies[detid_lut[detid]] for detid in detid_isolated]
    )
    hlvs["isolated_E_fraction"] = hlvs["isolated_energy"] / hlvs["sum_energy"]
    for var in ["x", "y", "z"]:
        var_weighted = static_feature_matrix[:, 1] * hit_energies
        mean = np.mean(var_weighted)
        hlvs[var + "_mean"] = mean
        hlvs[var + "_std"] = np.std(var_weighted)
        hlvs[var + "_mom3"] = np.power(
            np.sum(np.power(var_weighted - mean, 3)), 1 / 3
        )

    # yappi.stop()
    # current_module = sys.modules[__name__]
    # yappi.get_func_stats(
    #     filter_callback=lambda x: yappi.module_matches(x, [current_module])
    # ).sort("name", "desc").print_all()

    # Build the graph
    graph = torch_geometric.data.Data(
        x=torch.tensor(hit_energies, dtype=torch.float).reshape((num_nodes, 1)),
        y=torch.tensor(event[conf.loader.braches.energy][0], dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )

    graph.feature_mtx_static = torch.tensor(
        np.asarray(static_feature_matrix, dtype=np.float32), dtype=torch.float
    ).reshape((num_nodes, -1))

    graph.hlvs = torch.tensor(
        [hlvs[k] for k in conf.loader.hlvs], dtype=torch.float
    ).reshape((1, -1))

    return graph
