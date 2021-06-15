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

    return batch


# for contype in ["inner", "forward", "backward"]:
#     for ilayer in range(conf.nlayers):

#         a = getattr(batch, f"new_{contype}_edges_per_layer")[ilayer]
#         b = getattr(batch, f"{contype}_edges_per_layer")[ilayer]
#         if a.shape == b.shape:
#             if torch.all(a == b):
#                 continue
#         print(f"Error for {contype} - layer {ilayer}")
#         print("\t", a.shape, b.shape)

# mask_1stlayer = (batch.layers == 0) & (batch.batch == 0)
# ei_mask = mask_1stlayer[batch.edge_index[0]] & mask_1stlayer[batch.edge_index[1]]
