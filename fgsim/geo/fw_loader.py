# %%
import numpy as np
import h5py as h5
import os
import torch
from ..config import conf
from ..utils.logger import logger
from torch_geometric.data import Data


# %%

if os.path.isfile(conf["graphpath"]):
    lD = torch.load(conf["graphpath"])
    for key in lD:
        exec(f"{key}=lD['{key}']")
    logger.info("Done loading graph.")
else:
    logger.info("Generating graph...")
    from ..fw_data_loader import data_generator

    arr = np.zeros(next(data_generator()).shape)

    num_nodes = arr.shape[0] * arr.shape[1] * arr.shape[2]
    # Node features:
    # 0.Energy
    # 1.layer
    # 2-5 hidden
    num_node_features = 6

    num_edge_features = 1
    # Node feature matrix
    x = np.empty((0, num_node_features), dtype=int)
    # edge_index
    edge_index = np.empty((2, 0), dtype=int)
    # save the type of connection
    edge_attr = np.empty((0, num_edge_features), dtype=int)

    def arrpos(i, j, k, shape):
        return i * shape[1] * shape[2] + j * (shape[2]) + (k + 1)

    nlayers = arr.shape[0]
    # Keep track of connection per layer
    # layer node1 node2
    edges_per_layer = np.empty((nlayers, 2, 0), dtype=int)
    forward_edges_per_layer = np.empty((nlayers, 2, 0), dtype=int)
    backward_edges_per_layer = np.empty((nlayers, 2, 0), dtype=int)
    # layer node
    nodes_per_layer = np.empty((nlayers, 1, 0), dtype=int)

    # layer
    num_edges = 0
    for i in range(arr.shape[0]):
        nodefeaturearr = np.array([[0, i] + [0 for _ in range(4)]])
        # row
        for j in range(arr.shape[1]):
            # column
            for k in range(arr.shape[2]):
                curid = arrpos(i, j, k, arr.shape)
                x = np.append(x, nodefeaturearr, axis=0)
                nodes_per_layer[i] = np.append(
                    nodes_per_layer[i],
                    np.array([[curid]]),
                    axis=1,
                )
                # print(f"{curid}: {i} {j} {k} | {arr.shape}")
                # regular neighbors
                if j != 0:
                    targetid = arrpos(i, j - 1, k, arr.shape)
                    edge_index = np.append(
                        edge_index,
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edges_per_layer[i] = np.append(
                        edges_per_layer[i],
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edge_attr = np.append(edge_attr, np.array([[0]]), axis=0)

                    num_edges = num_edges + 1
                if j != arr.shape[1] - 1:
                    targetid = arrpos(i, j + 1, k, arr.shape)
                    edge_index = np.append(
                        edge_index,
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edges_per_layer[i] = np.append(
                        edges_per_layer[i],
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edge_attr = np.append(edge_attr, np.array([[0]]), axis=0)
                    num_edges = num_edges + 1
                if k != 0:
                    targetid = arrpos(i, j, k - 1, arr.shape)
                    edge_index = np.append(
                        edge_index,
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edges_per_layer[i] = np.append(
                        edges_per_layer[i],
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edge_attr = np.append(edge_attr, np.array([[0]]), axis=0)
                    num_edges = num_edges + 1
                if k != arr.shape[1] - 1:
                    targetid = arrpos(i, j, k + 1, arr.shape)
                    edge_index = np.append(
                        edge_index,
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edges_per_layer[i] = np.append(
                        edges_per_layer[i],
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edge_attr = np.append(edge_attr, np.array([[0]]), axis=0)
                    num_edges = num_edges + 1
                # forward
                if i != arr.shape[0] - 1:
                    targetid = arrpos(i + 1, j, k, arr.shape)
                    edge_index = np.append(
                        edge_index,
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    forward_edges_per_layer[i] = np.append(
                        forward_edges_per_layer[i],
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edge_attr = np.append(edge_attr, np.array([[1]]), axis=0)
                    num_edges = num_edges + 1
                # backward
                if i != 0:
                    targetid = arrpos(i - 1, j, k, arr.shape)
                    edge_index = np.append(
                        edge_index,
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    backward_edges_per_layer[i] = np.append(
                        backward_edges_per_layer[i],
                        np.array([[curid], [targetid]]),
                        axis=1,
                    )
                    edge_attr = np.append(edge_attr, np.array([[2]]), axis=0)
                    num_edges = num_edges + 1

    # %%
    graph = Data(
        x=torch.tensor(x, dtype=torch.int),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.int),
    )
    # %%
    num_edges = graph.edge_attr.shape[0]
    assert tuple(graph.x.shape) == (num_nodes, num_node_features)
    assert tuple(graph.edge_attr.shape) == (num_edges, num_edge_features)
    assert tuple(graph.edge_index.shape) == (2, num_edges)

    torch.save(
        {
            "graph": graph,
            "nodes_per_layer": nodes_per_layer,
            "edges_per_layer": edges_per_layer,
            "forward_edges_per_layer": forward_edges_per_layer,
            "backward_edges_per_layer": backward_edges_per_layer,
        },
        conf["graphpath"],
    )
    logger.info("Done generating graph.")

num_nodes, num_node_features = tuple(graph.x.shape)
num_edges, num_edge_features = tuple(graph.edge_attr.shape)

print("foo")
# edges = graph.edge_index.T

# def f(edge):
#     return edge.flip(0) not in edges, edge

# # from multiprocessing import Pool
# # with Pool(5) as p:
# #     for b, edge in p.imap(f, edges):
# #         if b:
# #             print(edge)

# for b, edge in map(f, edges):
#     if b:
#         print(edge)
