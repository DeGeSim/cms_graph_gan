import torch

from fgsim.io.batch_tools import (
    pcs_to_batch_reshape_direct,
    pcs_to_batch_reshape_list,
    pcs_to_batch_sort_direct,
    pcs_to_batch_sort_list,
    pcs_to_batch_v1,
)
from fgsim.utils.timeit import timeit

device = torch.device("cpu")


def measure_pcs_to_batch():
    n_graphs = 50
    # n_features = 3
    events = torch.arange(0, n_graphs).repeat(1000)
    # pcs = torch.rand((n_graphs * 10, n_features))
    pcs = torch.stack([events, events + 0.3, events + 0.6]).T

    iterations = 100

    functions = {
        "pcs_to_batch_v1": pcs_to_batch_v1,
        "pcs_to_batch_reshape_direct": pcs_to_batch_reshape_direct,
        "pcs_to_batch_reshape_list": pcs_to_batch_reshape_list,
        "pcs_to_batch_sort_direct": pcs_to_batch_sort_direct,
        "pcs_to_batch_sort_list": pcs_to_batch_sort_list,
    }
    for name, fct in functions.items():
        t = timeit(fct, n=iterations)(pcs, events)
        print(f"Time for {name} in {iterations} iterations: {t.time}")


if __name__ == "__main__":
    measure_pcs_to_batch()
