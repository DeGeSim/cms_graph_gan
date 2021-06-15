import torch

from ..config import conf
from ..utils.checkgraph import checkgraph
from ..utils.logger import logger


def split_layer_subgraphs(batch):
    # provide masks for the layers
    batch.layermask = torch.vstack(
        [batch.layers == ilayer for ilayer in range(conf.nlayers)]
    )

    batch.inner_edges_per_layer = []
    batch.forward_edges_per_layer = []
    batch.backward_edges_per_layer = []

    batch.mask_node_inner_per_layer = []
    batch.mask_node_forward_per_layer = []
    batch.mask_node_backward_per_layer = []
    for ilayer in range(conf.nlayers):
        # Create the mask for the nodes in the current/next/previous layer
        mask_cur_layer = batch.layermask[ilayer]
        if ilayer == 0:
            mask_previous_layer = torch.zeros(mask_cur_layer.shape, dtype=torch.bool)
        else:
            mask_previous_layer = batch.layermask[ilayer - 1]
        if ilayer == conf.nlayers - 1:
            mask_next_layer = torch.zeros(mask_cur_layer.shape, dtype=torch.bool)
        else:
            mask_next_layer = batch.layermask[ilayer + 1]

        mask_node_inner, edge_index_inner = edge_index_on_subgraph(
            batch.edge_index, mask_cur_layer, mask_cur_layer
        )
        mask_node_forward, edge_index_forward = edge_index_on_subgraph(
            batch.edge_index, mask_cur_layer, mask_next_layer
        )
        mask_node_backward, edge_index_backwards = edge_index_on_subgraph(
            batch.edge_index,
            mask_cur_layer,
            mask_previous_layer,
        )

        batch.inner_edges_per_layer.append(edge_index_inner)
        batch.forward_edges_per_layer.append(edge_index_forward)
        batch.backward_edges_per_layer.append(edge_index_backwards)

        batch.mask_node_inner_per_layer.append(mask_node_inner)
        batch.mask_node_forward_per_layer.append(mask_node_forward)
        batch.mask_node_backward_per_layer.append(mask_node_backward)

    checkgraph(batch)

    return batch


def edge_index_on_subgraph(edge_index, node_mask_from, node_mask_to):
    # filter the nodes
    node_mask = node_mask_from | node_mask_to

    # generate a mask for the edge index
    mask_edge_index = node_mask_from[edge_index[0]] & node_mask_to[edge_index[1]]

    # filter the edge index
    filtered_edge_index = edge_index[:, mask_edge_index]

    # provide a mapping that allows mapping the edge indices from the full batch
    # to the batch reduced to the layer-subgraphs
    # node_index_to_subgraph_index_map = torch.cumsum(node_mask)
    # edge_index = node_index_to_subgraph_index_map[edge_index]

    return (node_mask, filtered_edge_index)


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
