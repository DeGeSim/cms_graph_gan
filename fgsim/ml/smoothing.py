import matplotlib.pyplot as plt
import numpy as np
import torch

from fgsim.config import conf


def smooth_features(x):
    # for idx, name in enumerate(conf.loader.cell_prop_keys):
    #     plotdist(name, x[:, idx])
    x = (
        x
        + torch.from_numpy(
            np.random.multivariate_normal(
                [0.0, 0.0, 0.0, 0.0],
                np.diag(conf.training.smoothing_vars),
                conf.loader.max_points * conf.loader.batch_size,
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