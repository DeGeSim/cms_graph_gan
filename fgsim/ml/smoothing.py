import matplotlib.pyplot as plt
import numpy as np
import torch

from fgsim.config import conf


def turnoff(step):
    if conf.training.smoothing.decay == 0:
        return 1
    step *= conf.training.smoothing.decay
    if step > np.pi:
        return 0
    else:
        return (np.cos(step) + 1) / 2


def smooth_features(x, step):
    # for idx, name in enumerate(conf.loader.cell_prop_keys):
    #     plotdist(name, x[:, idx])
    x = (
        x
        + torch.from_numpy(
            np.random.multivariate_normal(
                [0.0] * conf.loader.n_features,
                np.diag(conf.training.smoothing.vars) * turnoff(step),
                size=x.shape[:-1],
            )
        ).float()
    )
    return x
    # for idx, name in enumerate(conf.loader.cell_prop_keys):
    #     plotdist(name, x[:, idx], presmooth=False)


def plotdist(name, arr, presmooth=True):
    dists = (
        torch.cdist(arr[:5000].reshape(1, -1, 1), arr[:5000].reshape(1, -1, 1))
        .reshape(-1)
        .numpy()
    )
    fig, ax = plt.subplots()
    ax.hist(dists, bins=300)
    fig.savefig(
        f"wd/{name}-dists.png" if presmooth else f"wd/{name}-dists_smoothed.png"
    )
    plt.close()
