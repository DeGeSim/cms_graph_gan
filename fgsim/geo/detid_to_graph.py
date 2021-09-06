"""
Conversion from list of hit to graph
"""

# import sys

import awkward as ak
import numpy as np
import pandas as pd
import torch
import torch_geometric

# import yappi
from torch_geometric.data import Data as GraphType

from fgsim.config import conf

# with uproot.open(conf.path.geo_lup) as rf:
#     geo_lup = rf['analyzer/tree;1'].arrays(library='ak')
# geo_lup = ak.to_pandas(geo_lup)
# geo_lup.set_index('globalid', inplace=True)
# geo_lup.to_pickle('data/hgcal/DetIdLUT_full.pd')

geo_lup = pd.read_pickle("data/hgcal/DetIdLUT_full.pd")


def event_to_graph(event: ak.highlevel.Record) -> GraphType:
    # yappi.start()

    # plt.close()
    # plt.cla()
    # plt.clf()
    # event_hitenergies = ak.to_numpy(event.rechit_energy)
    # event_hitenergies = event_hitenergies[event_hitenergies < 0.04]
    # plt.hist(event_hitenergies, bins=100)
    # plt.savefig("hist.png")
    # print("foo")

    # cut the hits 81% energy 56% of the hits @ 57 GeV
    energyfilter = event["rechit_energy"] > 0.0048
    num_nodes = sum(energyfilter)
    event_hitenergies = ak.to_numpy(event["rechit_energy"][energyfilter])
    event_detids = ak.to_numpy(event[conf.loader.id_key][energyfilter])
    # print(
    #     (
    #         sum(event_hitenergies) / sum(event["rechit_energy"]),
    #         len(event_hitenergies) / len(event["rechit_energy"]),
    #     )
    # )

    # Filter out the rows/detids that are not in the event
    geo_lup_filtered = geo_lup.loc[event_detids]
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

    static_feature_keys = [
        "x",
        "y",
        "z",
        "celltype",
        "issilicon",
    ]
    # compute static features
    static_feature_matrix = geo_lup_filtered[static_feature_keys].values

    # number detids in events
    detid_lut = {det_id: i for i, det_id in enumerate(event_detids)}

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
    eventid_set = set(event_detids)
    target_detid_in_event_mask = np.array(
        [(lambda x: x in eventid_set)(x) for x in edge_index_detid[1]]
    )
    edge_index_detid = edge_index_detid[:, target_detid_in_event_mask]

    # shift from detids to nodenumbers
    edge_index = np.vectorize(detid_lut.get)(edge_index_detid)

    # yappi.stop()
    # current_module = sys.modules[__name__]
    # yappi.get_func_stats(
    #     filter_callback=lambda x: yappi.module_matches(x, [current_module])
    # ).sort("name", "desc").print_all()

    graph = torch_geometric.data.Data(
        x=torch.tensor(event_hitenergies, dtype=torch.float32).reshape(
            (num_nodes, 1)
        ),
        y=event["gen_energy"],
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )

    graph.static_features = torch.tensor(
        np.asarray(static_feature_matrix, dtype=np.float32), dtype=torch.float32
    ).reshape((num_nodes, -1))
    return graph
