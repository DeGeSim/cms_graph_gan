import torch

from fgsim.utils.timeit import timeit

device = torch.device("cpu")

# fgsim.models.subnetworks.gen_deeptree_pc.branching
def reshape_features(
    mtx: torch.Tensor,
    n_parents: int,
    n_events: int,
    n_branches: int,
    n_features: int,
):
    magic_list = [colmtx.split(n_features, dim=1) for colmtx in mtx.split(1, dim=0)]

    out_list = []
    for iparent in range(n_parents):
        for ibranch in range(n_branches):
            for ievent in range(n_events):
                row = ievent + iparent * n_events
                out_list.append(magic_list[row][ibranch])
    return torch.cat(out_list)


def reshape_features2(
    mtx: torch.Tensor,
    n_parents: int,
    n_events: int,
    n_branches: int,
    n_features: int,
):
    return (
        mtx.reshape(n_parents, n_events, n_features * n_branches)
        .transpose(1, 2)
        .reshape(n_parents * n_branches, n_features, n_events)
        .transpose(1, 2)
        .reshape(n_parents * n_branches * n_events, n_features)
    )


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
    n_events = 50
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

    iterations = 100

    functions = {
        "iterate_reshape": reshape_features,
        "reshape_reshape": reshape_features2,
    }
    for name, fct in functions.items():
        t = timeit(fct, n=iterations)(**args)
        print(f"Time for {name} in {iterations} iterations: {t.time}")
        t_jitted = timeit(fct, n=iterations)(**args)
        print(f"Time for {name} in {iterations} iterations jitted: {t_jitted.time}")


if __name__ == "__main__":
    measure_reshape_features()
