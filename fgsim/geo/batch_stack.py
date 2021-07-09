import torch

from ..config import conf


def split_layer_subgraphs(batch):
    # provide masks for the layers
    batch.layermask = torch.vstack(
        [batch.layers == ilayer for ilayer in range(conf.nlayers)]
    )

    batch.inner_edges_per_layer = []
    batch.forward_edges_per_layer = []
    batch.backward_edges_per_layer = []

    batch.mask_inp_innerL = []
    batch.mask_inp_forwardL = []
    batch.mask_inp_backwardL = []

    batch.mask_outp_innerL = []
    batch.mask_outp_forwardL = []
    batch.mask_outp_backwardL = []

    for ilayer in range(conf.nlayers):
        # Create the mask for the nodes in the current/next/previous layer
        mask_cur_layer = batch.layermask[ilayer]
        if ilayer == 0:
            mask_previous_layer = torch.zeros(
                mask_cur_layer.shape, dtype=torch.bool
            )
        else:
            mask_previous_layer = batch.layermask[ilayer - 1]
        if ilayer == conf.nlayers - 1:
            mask_next_layer = torch.zeros(mask_cur_layer.shape, dtype=torch.bool)
        else:
            mask_next_layer = batch.layermask[ilayer + 1]

        # The masks for projecting out the relevant rows of the
        # feature array and the edge index that is shifted to this selection.
        (
            mask_input_inner,
            mask_output_inner,
            edge_index_inner,
        ) = edge_index_on_subgraph(batch.edge_index, mask_cur_layer, mask_cur_layer)
        (
            mask_input_forward,
            mask_output_forward,
            edge_index_forward,
        ) = edge_index_on_subgraph(
            batch.edge_index, mask_cur_layer, mask_next_layer
        )
        (
            mask_input_backward,
            mask_output_backward,
            edge_index_backward,
        ) = edge_index_on_subgraph(
            batch.edge_index,
            mask_cur_layer,
            mask_previous_layer,
        )

        # if not 0 in edge_index_inner.shape:
        #     logger.warn(
        #         (torch.min(edge_index_inner),
        #         torch.max(edge_index_inner),
        #         torch.sum(mask_input_inner),)
        #     )
        # if not 0 in edge_index_forward.shape:
        #     logger.warn(
        #         (torch.min(edge_index_forward),
        #         torch.max(edge_index_forward),
        #         torch.sum(mask_input_forward),)
        #     )
        # if not 0 in edge_index_backward.shape:
        #     logger.warn(
        #         (torch.min(edge_index_backward),
        #         torch.max(edge_index_backward),
        #         torch.sum(mask_input_backward),)
        #     )

        batch.inner_edges_per_layer.append(edge_index_inner)
        batch.forward_edges_per_layer.append(edge_index_forward)
        batch.backward_edges_per_layer.append(edge_index_backward)

        batch.mask_inp_innerL.append(mask_input_inner)
        batch.mask_inp_forwardL.append(mask_input_forward)
        batch.mask_inp_backwardL.append(mask_input_backward)

        batch.mask_outp_innerL.append(mask_output_inner)
        batch.mask_outp_forwardL.append(mask_output_forward)
        batch.mask_outp_backwardL.append(mask_output_backward)

    return batch


def edge_index_on_subgraph(edge_index, node_mask_from, node_mask_to):
    # example:
    # x=[[0], [1], [2], [3]]
    # edge_index= [
    #     [0, 1, 1, 2, 2, 3],
    #     [1, 0, 2, 1, 3, 2],
    # ]
    # node_mask_from = [F,T,T,F]
    # node_mask_to = [F,F,F,T]
    # filter the nodes
    node_mask_input = node_mask_from | node_mask_to
    # node_mask_input = [F,T,T,T]

    if torch.all(~node_mask_input):
        return (
            node_mask_input,
            torch.tensor([], dtype=torch.bool),
            torch.tensor([[], []], dtype=torch.int64),
        )

    # generate a mask for the edge index
    mask_edge_index = node_mask_from[edge_index[0]] & node_mask_to[edge_index[1]]
    # mask_edge_index = [F,F,F,F,T,F]

    # filter the edge index
    filtered_edge_index = edge_index[:, mask_edge_index]
    # filtered_edge_index= [
    #     [2],
    #     [3],
    # ]

    # Provide a mapping for the filtered_edge_index
    # to adress the reduced graph.

    # Just directly shifting the edge index leads to nodes from different
    # graphs mixing, as we do not select a continuous region.
    # To avoid this we apply a this using the cummulative sum:
    # Count the 1s in the mask with cumulative sum
    ei_map = torch.cumsum(node_mask_input, dim=0)
    # ei_map = [0,1,2,3]

    # Now this map provides the identity operation:
    # filtered_edge_index == ei_map[filtered_edge_index]
    # [[2],[3]] == [0,1,2,3][[[2],[3]]]

    # the first element selected by node_mask_input should be mapped to 0:

    ei_shift = ei_map[node_mask_input][0]
    # ei_shift = [0,1,2,3]
    ei_map = ei_map - ei_shift
    # ei_map = [-1,0,1,2]

    shifted_edge_index = ei_map[filtered_edge_index]
    # [[1],[2]] == [-1,0,1,2][[[2],[3]]]
    # This is the desired output, as the the new 0th node (old 1st)
    # is not connected, but still in the grap, the new 1st (old 2nd)
    #  is connected to the 2nd (old 3rd).
    if shifted_edge_index.shape[1] > 0:
        assert 0 <= torch.min(shifted_edge_index)
        assert (
            torch.max(shifted_edge_index) <= torch.sum(node_mask_input.long()) - 1
        )

    # Once the convolution  is completed, the output has len
    # sum(mask_node). To assign this output to the feature matrix,
    # it needs to be  masked. The mask needs to have as many elements
    # as the mask_node has ones and project out the entries of the
    # output that have been in the 'to' mask.
    # The nodes that correspond to the ones in the target layer
    # must be selected from the filtered output.
    node_mask_output = node_mask_to[node_mask_input]
    # node_mask_output = [F,F,F,T][[F,T,T,T]] = [F,F,T]
    # x[node_mask_input][node_mask_output]
    # = [0,1,2,3][[F,T,T,T]][[F,F,T]]
    # = [1,2,3][F,F,T] = [3] <- This is the node that is assiged.

    return (node_mask_input, node_mask_output, shifted_edge_index)


# # Test
# # Stack Graph 0--1--2--3
# # In layers   0  1  1  2

# g = torch_geometric.data.Data(
#     x=torch.tensor([[1], [2], [3], [4]]),
#     edge_index=torch.tensor(
#         [
#             [0, 1, 1, 2, 2, 3],
#             [1, 0, 2, 1, 3, 2],
#         ]
#     ),
# )
# g.layers = torch.tensor([0, 1, 1, 2])

# g = split_layer_subgraphs(g)

# # check the masks
# for ilayer in range(conf.nlayers):
#     for direction in ("forward", "inner", "backward"):
#         for in_or_out in ("inp", "outp"):
#             mask = getattr(g, f"mask_{in_or_out}_{direction}L")[ilayer]
#             if in_or_out == "inp":
#                 if direction == "inner":
#                     assert torch.equal(mask, g.layers == ilayer)
#                 if direction == "forward":
#                     assert torch.equal(
#                         mask, (g.layers == ilayer) | (g.layers == ilayer + 1)
#                     )
#                 if direction == "backward":
#                     assert torch.equal(
#                         mask, (g.layers == ilayer) | (g.layers == ilayer - 1)
#                     )
#             if in_or_out == "outp":
#                 if direction == "inner":
#                     assert torch.equal(mask, (g.layers == ilayer)[(g.layers == ilayer)])
#                 if direction == "forward":
#                     assert torch.equal(
#                         mask,
#                         (g.layers == ilayer + 1)[
#                             (g.layers == ilayer) | (g.layers == ilayer + 1)
#                         ],
#                     )
#                 if direction == "backward":
#                     assert torch.equal(
#                         mask,
#                         (g.layers == ilayer - 1)[
#                             (g.layers == ilayer - 1) | (g.layers == ilayer)
#                         ],
#                     )


# empty_edge_index = torch.tensor([[], []], dtype=torch.int64)

# assert torch.equal(g.forward_edges_per_layer[0], torch.tensor([[0], [1]]))
# assert torch.equal(g.forward_edges_per_layer[1], torch.tensor([[1], [2]]))
# assert torch.equal(g.forward_edges_per_layer[2], empty_edge_index)

# assert torch.equal(g.inner_edges_per_layer[0], empty_edge_index)
# assert torch.equal(g.inner_edges_per_layer[1], torch.tensor([[0, 1], [1, 0]]))
# assert torch.equal(g.inner_edges_per_layer[2], empty_edge_index)

# assert torch.equal(g.backward_edges_per_layer[0], empty_edge_index)
# assert torch.equal(g.backward_edges_per_layer[1], torch.tensor([[1], [0]]))
# assert torch.equal(g.backward_edges_per_layer[2], torch.tensor([[2], [1]]))

# assert g.layermask.long().sum() == len(g.x)
