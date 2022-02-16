import torch

from fgsim.io.pcgraph_seq.aggr_and_sort_points import aggr_and_sort_points


def test_aggr_and_sort_points():
    def c_sort(a):
        return torch.sort(a, dim=0, descending=True, stable=True)[0]

    arr = torch.tensor(
        [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
        ]
    )
    arr_sort = c_sort(arr)
    res = torch.tensor(
        [
            [0, 0, 0, 3],
            [0, 0, 1, 2],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ]
    )
    res_sort = c_sort(res)

    assert torch.all(aggr_and_sort_points(arr_sort) == res_sort)
