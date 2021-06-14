import torch
import torch_geometric


def stack_batch_edge_indexes(batch):
    res = []
    # We want to have a adjectentcy matrix in the from of
    # batch.forward_edges_per_layer[ilayer] available in
    # the same stacked format as the full adjectency matrix.

    # batch.forward_edges_per_layer is at this point
    # batch_size x nlayer as list of a list.
    # We need to join them for each of the sample,
    # so we need to repack them to nlayer x batch_size.
    layer_adj_list = list(zip(*batch.inner_edges_per_layer))

    # For a batch:
    # Join the edge indexes of the partial adjectency matrices
    # in the same way as for the proper adj matrix
    # -> add the start number of the batch to the indices
    # and stack them
    for layer_edge_index in layer_adj_list:
        sel_edge_index_in_layerL = [
            e + start  # shift the index of the nodes up
            # by the number of nodes of the already added graphs
            for e, start in zip(layer_edge_index, batch.ptr[:-1])
            if e.shape != torch.Size([0])  # remove empty layers
        ]
        if sel_edge_index_in_layerL != []:
            res.append(torch.hstack(sel_edge_index_in_layerL))
        else:
            res.append(torch.empty((2, 0), dtype=torch.int64))
    batch.inner_edges_per_layer = res

    res = []
    for layer_edge_index in zip(*batch.forward_edges_per_layer):
        sel_edge_index_in_layerL = [
            e + start  # shift the index of the nodes up
            # by the number of nodes of the already added graphs
            for e, start in zip(layer_edge_index, batch.ptr[:-1])
            if e.shape != torch.Size([0])  # remove empty layers
        ]
        if sel_edge_index_in_layerL != []:
            res.append(torch.hstack(sel_edge_index_in_layerL))
        else:
            res.append(torch.empty((2, 0), dtype=torch.int64))
    batch.forward_edges_per_layer = res

    res = []
    for layer_edge_index in zip(*batch.backward_edges_per_layer):
        sel_edge_index_in_layerL = [
            e + start  # shift the index of the nodes up
            # by the number of nodes of the already added graphs
            for e, start in zip(layer_edge_index, batch.ptr[:-1])
            if e.shape != torch.Size([0])  # remove empty layers
        ]
        if sel_edge_index_in_layerL != []:
            res.append(torch.hstack(sel_edge_index_in_layerL))
        else:
            res.append(torch.empty((2, 0), dtype=torch.int64))
    batch.backward_edges_per_layer = res

    # assert (
    #     sum(
    #         [
    #             sum([len(e_idx.T) for e_idx in llidx])
    #             for llidx in (
    #                 batch.forward_edges_per_layer,
    #                 batch.backward_edges_per_layer,
    #                 batch.inner_edges_per_layer,
    #             )
    #         ]
    #     )
    #     == len(batch.edge_index.T)
    # )

    return batch.cpu()


# %%
# The test
# Make sure the edge_index s for the layer are joined in the same way
# x = torch.tensor([[1], [2], [3], [4]])
# edge_index = torch.tensor([[0, 0, 0, 1], [1, 2, 3, 2]])
# grap = torch_geometric.data.Data(x, edge_index)

# grap.inner_edges_per_layer = [edge_index for _ in range(3)]
# grap.forward_edges_per_layer = [edge_index for _ in range(2)]
# grap.backward_edges_per_layer = [edge_index for _ in range(1)]

# batch = batch.from_data_list((grap, grap, grap))

# batch = stack_batch_edge_indexes(batch)
# assert torch.all(batch.inner_edges_per_layer[0] == batch.edge_index)
# assert len(grap.inner_edges_per_layer)==3
# assert len(grap.forward_edges_per_layer)==2
# assert len(grap.backward_edges_per_layer)==1
# assert torch.all(batch.forward_edges_per_layer[0] == batch.edge_index)
# assert torch.all(batch.backward_edges_per_layer[0] == batch.edge_index)
