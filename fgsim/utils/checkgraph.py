import torch

from ..config import conf

# Collection of functions to check if inner/forward/backward
# edges have been batched correctly.


def filter_edge_index_for_graph(batch, edge_index, igraph=0):
    return edge_index[
        :,
        (batch.ptr[igraph] <= edge_index[0, :])
        & (edge_index[0, :] < batch.ptr[igraph + 1]),
    ]


def change_ptrs(x):
    res = [0]
    i = 0
    cur = x[0]
    while i < len(x):
        if cur != x[i]:
            res.append(i)
            cur = x[i]
        i += 1
    res.append(i)
    return torch.tensor(res)


# get the vector giving the start and stop position for
# each layer in the batch
#  nlayer+1 elements, starts at 0
# aquivalent for layers what batch.ptr is for batches
def get_layer_ptr(batch, igraph=0):
    # Slice the layer vector with for the current graph
    layer_of_graph = batch.layers[batch.ptr[igraph] : batch.ptr[igraph + 1]]
    res = change_ptrs(layer_of_graph)
    return res + batch.ptr[igraph]


def checkgraph(batch, igraph=0):
    layerpos = get_layer_ptr(batch, igraph)
    # inner
    for ilayer in range(conf.nlayers):
        edge_index = batch.inner_edges_per_layer[ilayer]
        # filter for the graph
        edge_index = filter_edge_index_for_graph(batch, edge_index, igraph)
        # make sure the layer edges are within one layer
        assert torch.all(
            (layerpos[ilayer] <= edge_index) & (edge_index < layerpos[ilayer + 1])
        )

    for ilayer in range(conf.nlayers):
        edge_index = batch.forward_edges_per_layer[ilayer]
        # filter for the graph
        edge_index = filter_edge_index_for_graph(batch, edge_index, igraph)

        # forward_edges_per_layer[i] map i->i+1 last layer empty
        if ilayer < conf.nlayers - 1:
            # check that the edges map from layer = ilayer
            assert torch.all(
                (layerpos[ilayer] <= edge_index[0, :])
                & (edge_index[0, :] < layerpos[ilayer + 1])
            )
            # ... to ilayer+1
            assert torch.all(
                (layerpos[ilayer + 1] <= edge_index[1, :])
                & (edge_index[1, :] < layerpos[ilayer + 2])
            )

    for ilayer in range(conf.nlayers):
        edge_index = batch.backward_edges_per_layer[ilayer]
        # filter for the graph
        edge_index = filter_edge_index_for_graph(batch, edge_index, igraph)

        # forward_edges_per_layer[i] map i->i+1 last layer empty
        if ilayer > 0:
            # check that the edges map from layer = ilayer
            assert torch.all(
                (layerpos[ilayer] <= edge_index[0, :])
                & (edge_index[0, :] < layerpos[ilayer + 1])
            )
            # to layer = ilayer-1
            assert torch.all(
                (layerpos[ilayer - 1] <= edge_index[1, :])
                & (edge_index[1, :] < layerpos[ilayer])
            )
