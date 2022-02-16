import torch

from fgsim.models.subnetworks.gen_deeptree_pc.branching import reshape_features
from fgsim.utils.timeit import timeit

device = torch.device("cpu")

# Test the reshaping
def demo_mtx(*, n_parents, n_events, n_branches, n_features):
    mtx = torch.ones(n_parents * n_events, n_branches * n_features)
    i, j = 0, 0
    ifield = 0
    for iparent in range(n_parents):
        for ibranch in range(n_branches):
            for ievent in range(n_events):
                for ifeature in range(n_features):
                    # print(f"Acessing {i}, {j+ifeature}")
                    mtx[i, ifeature + ibranch * n_features] = ifield
                ifield = ifield + 1
                i = i + 1
            i = i - n_events
            j = j + 1
        i = i + n_events
        j = j - n_branches
    return mtx


def measure_reshape_features():
    n_parents = 100
    n_events = 5
    n_branches = 2
    n_features = 10
    mtx = demo_mtx(
        n_parents=n_parents,
        n_events=n_events,
        n_branches=n_branches,
        n_features=n_features,
    )
    args = {
        "mtx": mtx,
        "n_parents": n_parents,
        "n_events": n_events,
        "n_branches": n_branches,
        "n_features": n_features,
    }

    iterations = 10000
    jitted = torch.jit.script(reshape_features)

    t_vanilla = timeit(reshape_features, n=iterations)(**args)
    t_jitted = timeit(jitted, n=iterations)(**args)
    print(
        f"Time for {iterations} iterations: vanilla {t_vanilla.time} jitted"
        f" {t_jitted.time} speedup {t_vanilla.time/t_jitted.time} "
    )


if __name__ == "__main__":
    measure_reshape_features()
