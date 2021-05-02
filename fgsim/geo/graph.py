import numpy as np
import torch

# Node features:
# 0.Energy
# 1-4 hidden
num_node_features = 3
num_edge_features = 1


def arrpos(i, j, k, shape):
    return i * shape[1] * shape[2] + j * (shape[2]) + k


# Check for this function
# np.array(
#     [
#         [[arrpos(i, j, k, (2, 3, 4)) for k in range(4)] for j in range(3)]
#         for i in range(2)
#     ]
# ) == np.arange(2 * 3 * 4).reshape(2, 3, 4)


def getNonZeroIdxs(arr):
    return torch.arange(len(arr))[arr != 0.0]


def grid_to_graph(caloimg):
    caloimg = np.swapaxes(caloimg, 0, 2)
    nlayers, nrows, ncolumns = caloimg.shape

    # Node feature matrix
    feature_mtx = []
    # layer node
    nodes_per_layer = [[] for _ in range(nlayers)]

    # Save the edges
    # adj_mtx_coo
    adj_mtx_coo = []
    # Keep track of connection per layer
    # layer node1 node2
    inner_edges_per_layer = [[] for _ in range(nlayers)]
    forward_edges_per_layer = [[] for _ in range(nlayers)]
    backward_edges_per_layer = [[] for _ in range(nlayers)]

    # save the type of connection 0 inlayer 1 forward 2 backward
    edge_attr = []

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
                nodefeaturearr = np.zeros((num_node_features,))
                nodefeaturearr[0] = caloimg[ilayer, jrow, kcolumn]
                # extend the feature matrix
                feature_mtx.append(nodefeaturearr)
                nodes_per_layer[ilayer].append(nodefeaturearr)

                # regular neighbors
                # up if not top t
                if jrow != 0 and caloimg[ilayer, jrow - 1, kcolumn]:
                    targetid = arrpos(ilayer, jrow - 1, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    adj_mtx_coo.append(edge)
                    inner_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(0)

                    num_edges = num_edges + 1
                # down
                if jrow != caloimg.shape[1] - 1 and caloimg[ilayer, jrow + 1, kcolumn]:
                    targetid = arrpos(ilayer, jrow + 1, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    adj_mtx_coo.append(edge)
                    inner_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(0)
                    num_edges = num_edges + 1
                # left
                if kcolumn != 0 and caloimg[ilayer, jrow, kcolumn - 1]:
                    targetid = arrpos(ilayer, jrow, kcolumn - 1, caloimg.shape)
                    edge = np.array([curid, targetid])
                    adj_mtx_coo.append(edge)
                    inner_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(0)
                    num_edges = num_edges + 1
                # right
                if (
                    kcolumn != caloimg.shape[1] - 1
                    and caloimg[ilayer, jrow, kcolumn + 1]
                ):
                    targetid = arrpos(ilayer, jrow, kcolumn + 1, caloimg.shape)
                    edge = np.array([curid, targetid])
                    adj_mtx_coo.append(edge)
                    inner_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(0)
                    num_edges = num_edges + 1
                # forward
                if (
                    ilayer != caloimg.shape[0] - 1
                    and caloimg[ilayer + 1, jrow, kcolumn]
                ):
                    targetid = arrpos(ilayer + 1, jrow, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    adj_mtx_coo.append(edge)
                    forward_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(1)
                    num_edges = num_edges + 1
                # backward
                if ilayer != 0 and caloimg[ilayer - 1, jrow, kcolumn]:
                    targetid = arrpos(ilayer - 1, jrow, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    adj_mtx_coo.append(edge)
                    backward_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(2)
                    num_edges = num_edges + 1
                runningindex = runningindex + 1
    # the DetIds in the adjacency matrix need to be
    # tranformed to real indices of the feature matrix
    for i in range(num_edges):
        adj_mtx_coo[i][0] = globalid_to_indexD[adj_mtx_coo[i][0]]
        adj_mtx_coo[i][1] = globalid_to_indexD[adj_mtx_coo[i][1]]

    # The following is not needed, because for python
    # the manipulated edge is the same for all these matrices:
    # adj_mtx_coo[0] is inner_edges_per_layer[0][0] etc.
    # for ajdmtxi in (
    #     inner_edges_per_layer,
    #     forward_edges_per_layer,
    #     backward_edges_per_layer,
    # ):
    #     for ilayer in range(nlayers):
    #         for i in range(num_edges):
    #             ajdmtxi[ilayer][i][0] = globalid_to_indexD[ajdmtxi[ilayer][i][0]]
    #             ajdmtxi[ilayer][i][1] = globalid_to_indexD[ajdmtxi[ilayer][i][1]]

    return (
        torch.tensor(feature_mtx, dtype=torch.float32),
        torch.tensor(adj_mtx_coo, dtype=torch.int64),
        [torch.tensor(e,dtype=torch.int64) for e in inner_edges_per_layer],
        [torch.tensor(e,dtype=torch.int64) for e in forward_edges_per_layer],
        [torch.tensor(e,dtype=torch.int64) for e in backward_edges_per_layer],
    )


# import h5py as h5

# with h5.File("wd/forward/Ele_FixedAngle/EleEscan_1_1.h5") as f:
#     caloimgs = f["ECAL"][0:10]
# (
#     feature_mtx,  # nodes x num_node_features
#     adj_mtx_coo,  # 2 x num_edges
#     inner_edges_per_layer,  # num_layer Lists []
#     forward_edges_per_layer,
#     backward_edges_per_layer,
# ) = grid_to_graph(caloimgs[0])
