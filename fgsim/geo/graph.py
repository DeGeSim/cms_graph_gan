import numpy as np
import torch
from torch_geometric.data import Data

from ..config import conf


def arrpos(ilayer, irow, icolumn, shape):
    return ilayer * shape[1] * shape[2] + irow * (shape[2]) + icolumn


# Check for this arrpos
# np.array(
#     [
#         [[arrpos(i, j, k, (2, 3, 4)) for k in range(4)] for j in range(3)]
#         for i in range(2)
#     ]
# ) == np.arange(2 * 3 * 4).reshape(2, 3, 4)


def getNonZeroIdxs(arr):
    return torch.arange(len(arr))[arr != 0.0]


def grid_to_graph(caloimg):
    # columns rows layers -> layers rows columns
    caloimg = caloimg.T
    nlayers, nrows, ncolumns = caloimg.shape

    # Node feature matrix
    feature_mtx_dyn = []
    feature_mtx_static = []
    # layer node
    layers = []
    rows = []
    columns = []

    # Save the edges
    edge_index = []
    # layer
    num_edges = 0
    globalid_to_indexD = {}
    runningindex = 0
    for ilayer in range(nlayers):
        # row
        for jrow in range(nrows):
            # column
            for kcolumn in range(ncolumns):
                if caloimg[ilayer, jrow, kcolumn] == 0:
                    continue
                # get the position in the faltened array
                curid = arrpos(ilayer, jrow, kcolumn, caloimg.shape)
                # save the position of the node
                globalid_to_indexD[curid] = runningindex
                # contruct the feature array for the node

                # extend the feature matrix
                features_dyn = np.array([caloimg[ilayer, jrow, kcolumn]])
                feature_mtx_dyn.append(features_dyn)

                features_static = np.array(
                    [float(ilayer), float(jrow), float(kcolumn)]
                )
                feature_mtx_static.append(features_static)

                layers.append(ilayer)
                rows.append(jrow)
                columns.append(kcolumn)

                # regular neighbors
                # up if not top t
                if jrow != 0 and caloimg[ilayer, jrow - 1, kcolumn]:
                    targetid = arrpos(ilayer, jrow - 1, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    num_edges = num_edges + 1
                # down
                if (
                    jrow != caloimg.shape[1] - 1
                    and caloimg[ilayer, jrow + 1, kcolumn]
                ):
                    targetid = arrpos(ilayer, jrow + 1, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    num_edges = num_edges + 1
                # left
                if kcolumn != 0 and caloimg[ilayer, jrow, kcolumn - 1]:
                    targetid = arrpos(ilayer, jrow, kcolumn - 1, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    num_edges = num_edges + 1
                # right
                if (
                    kcolumn != caloimg.shape[1] - 1
                    and caloimg[ilayer, jrow, kcolumn + 1]
                ):
                    targetid = arrpos(ilayer, jrow, kcolumn + 1, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    num_edges = num_edges + 1
                # forward
                if (
                    ilayer != caloimg.shape[0] - 1
                    and caloimg[ilayer + 1, jrow, kcolumn]
                ):
                    targetid = arrpos(ilayer + 1, jrow, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    num_edges = num_edges + 1
                # backward
                if ilayer != 0 and caloimg[ilayer - 1, jrow, kcolumn]:
                    targetid = arrpos(ilayer - 1, jrow, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    num_edges = num_edges + 1
                runningindex = runningindex + 1
    # the DetIds in the adjacency matrix need to be
    # tranformed to real indices of the feature matrix
    for i in range(num_edges):
        edge_index[i][0] = globalid_to_indexD[edge_index[i][0]]
        edge_index[i][1] = globalid_to_indexD[edge_index[i][1]]

    feature_mtx_dyn = torch.tensor(feature_mtx_dyn, dtype=torch.float32).reshape(
        -1, 1
    )
    feature_mtx_static = torch.tensor(
        feature_mtx_static, dtype=torch.float32
    ).reshape(feature_mtx_dyn.shape[0], conf.model.static_features)
    edge_index = torch.tensor(edge_index, dtype=torch.int64).T.reshape(2, -1)

    graph = Data(x=feature_mtx_dyn, edge_index=edge_index)
    graph.feature_mtx_static = feature_mtx_static

    graph.layers = torch.tensor(layers, dtype=torch.int64)
    graph.nlayers = torch.tensor(nlayers, dtype=torch.int64)

    graph.rows = torch.tensor(rows, dtype=torch.int64)
    graph.nrows = torch.tensor(nrows, dtype=torch.int64)

    graph.columns = torch.tensor(columns, dtype=torch.int64)
    graph.ncolumns = torch.tensor(ncolumns, dtype=torch.int64)

    return graph.cpu()
