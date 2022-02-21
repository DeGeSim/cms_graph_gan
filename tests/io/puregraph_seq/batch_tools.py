import torch

from fgsim.io.batch_tools import (
    pcs_to_batch_reshape_direct,
    pcs_to_batch_reshape_list,
    pcs_to_batch_sort_direct,
    pcs_to_batch_sort_list,
    pcs_to_batch_v1,
)


def test_batch_from_pcs_list_reorder():
    n_graphs = 5
    # n_features = 3
    events = torch.arange(0, n_graphs).repeat(4)
    # pcs = torch.rand((n_graphs * 10, n_features))
    pcs = torch.stack([events, events + 0.3, events + 0.6]).T

    b1 = pcs_to_batch_v1(pcs, events)
    b2 = pcs_to_batch_reshape_direct(pcs, events)
    b3 = pcs_to_batch_reshape_list(pcs, events)
    b4 = pcs_to_batch_sort_direct(pcs, events)
    b5 = pcs_to_batch_sort_list(pcs, events)
    for igraph in range(n_graphs):
        assert ftx_eq(b1[igraph].x, b2[igraph].x)
        assert ftx_eq(b1[igraph].x, b3[igraph].x)
        assert ftx_eq(b1[igraph].x, b4[igraph].x)
        assert ftx_eq(b1[igraph].x, b5[igraph].x)


def ftx_eq(x1: torch.Tensor, x2: torch.Tensor) -> bool:
    idxs1 = x1[:, -1].sort().indices
    x1 = x1[idxs1]
    idxs2 = x2[:, -1].sort().indices
    x2 = x2[idxs2]
    if x1.shape != x2.shape:
        return False
    return bool(torch.all(x1 == x2))


# from fgsim.io.batch_tools import (
#     batch_from_pcs_list_events,
#     batch_from_pcs_list_magic,
#     batch_from_pcs_list_reorder,
# )

# def test_batch_from_pcs_list_reorder():
#     n_graphs = 5
#     n_features = 3
#     pcs = torch.rand((n_graphs * 10, n_features))
#
#     b1 = batch_from_pcs_list_events(pcs, events)
#     b2 = batch_from_pcs_list_reorder(pcs, events)
#     for igraph in range(n_graphs):
#         x1 = b1[igraph].x
#         x2 = b2[igraph].x
#         assert x1.shape == x2.shape
#         assert torch.all(x1 == x2)
