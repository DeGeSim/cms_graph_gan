import torch
from torch_geometric.data import Data

from ..config import conf
from ..geo.graph import grid_to_graph


def transform(inp) -> Data:
    data_dict = {k: v for k, v in zip(conf.loader.keylist, inp)}
    graph_ECAL = grid_to_graph(data_dict["ECAL"])
    graph_HCAL = grid_to_graph(data_dict["HCAL"])
    graph = merge_graphs(graph_ECAL, graph_HCAL)
    # Add the higher level variables to the grap
    for k in conf.loader.keylist:
        if k not in ["ECAL", "HCAL"]:
            setattr(graph, k, torch.tensor(data_dict[k]))
    return graph


def merge_graphs(ga: Data, gb: Data) -> Data:
    # shift all of the edge_indexes of the second graph
    gb.edge_index = gb.edge_index + len(ga.x)
    gb.layers = gb.layers + ga.nlayers

    # Compute the additional edges
    add_edge_index = []

    # Get the index of the elements in the first/last layer
    ga_last_layer_idxs = (ga.layers == ga.nlayers - 1).nonzero().reshape(-1)
    gb_first_layer_idxs = (gb.layers == ga.nlayers).nonzero().reshape(-1)

    if len(ga_last_layer_idxs) > 0 and len(gb_first_layer_idxs) > 0:
        for idx in ga_last_layer_idxs:
            # Calculate index of the row and the column in the target layer
            targetcol = torch.round(
                ga.columns[idx] / (ga.ncolumns - 1) * (gb.ncolumns - 1)
            )
            targetrow = torch.round(ga.rows[idx] / (ga.nrows - 1) * (gb.nrows - 1))

            # get the index of the the target cell within gb
            targetidxs_in_gb_first_layer_idxs = (
                (
                    (gb.columns[gb_first_layer_idxs] == targetcol)
                    & (gb.rows[gb_first_layer_idxs] == targetrow)
                )
                .nonzero()
                .reshape(-1)
            )

            assert len(targetidxs_in_gb_first_layer_idxs) in [0, 1]
            if len(targetidxs_in_gb_first_layer_idxs) == 1:
                # get the global index of the target with gb_first_layer_idxs
                targetidx = gb_first_layer_idxs[
                    targetidxs_in_gb_first_layer_idxs[0]
                ]
                add_edge_index.append([idx, targetidx])
                add_edge_index.append([targetidx, idx])

    connection_edge_index = torch.tensor(
        add_edge_index, dtype=torch.int64
    ).T.reshape(2, -1)

    assert not (ga.x.dim() != gb.x.dim() or ga.x.shape[1] != gb.x.shape[1])

    assert ga.edge_index.dim() == gb.edge_index.dim() == connection_edge_index.dim()
    assert (
        ga.edge_index.shape[0]
        == gb.edge_index.shape[0]
        == connection_edge_index.shape[0]
    )

    graph = Data(
        x=torch.vstack((ga.x, gb.x)),
        edge_index=torch.hstack(
            (
                ga.edge_index,
                connection_edge_index,
                gb.edge_index,
            )
        ),
    )

    assert not (
        ga.feature_mtx_static.dim() != gb.feature_mtx_static.dim()
        or ga.feature_mtx_static.shape[1] != gb.feature_mtx_static.shape[1]
    )

    graph.feature_mtx_static = torch.vstack(
        (ga.feature_mtx_static, gb.feature_mtx_static)
    )
    graph.layers = torch.hstack((ga.layers, gb.layers))

    return graph


# logger.info(f"""\
# Adding edge_index from column \
# {int(ga.columns[idx])}/{int(ga.ncolumns-1)} \
# ({float(ga.columns[idx]/(ga.ncolumns - 1) *100)}%)\
# -> ({int(gb.columns[targetidx])}/{int(gb.ncolumns-1)} \
# ({float(gb.columns[targetidx]/(gb.ncolumns - 1) *100)}%)\
# and row \
# {int(ga.rows[idx])}/{int(ga.nrows-1)} \
# ({float(ga.rows[idx]/(ga.nrows - 1) *100)}%)\
# -> ({int(gb.rows[targetidx])}/{int(gb.nrows-1)} \
# ({float(gb.rows[targetidx]/(gb.nrows - 1) *100)}%)\
# """)

# mtx = torch.ones((5,5))
# newmtx = torch.ones((2,2))
# idxs = mtx.nonzero()
# pos=torch.floor(idxs/torch.tensor(mtx.shape)*torch.tensor(newmtx.shape))
# logger.info('col split')
# logger.info(pos[:,0].reshape((*mtx.shape)))
# logger.info('row split')
# logger.info(pos[:,1].reshape((*mtx.shape)))
# pos.reshape((*mtx.shape,2))[0][-1] tensor([0., 1.])
# pos.reshape((*mtx.shape,2))[-1][-1] == tensor([1., 1.])
# pos.reshape((*mtx.shape,2))[-1][0] == tensor([1., 0.])
# pos.reshape((*mtx.shape,2))[0][0] == tensor([0., 0.])
