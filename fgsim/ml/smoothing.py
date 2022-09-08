import matplotlib.pyplot as plt
import numpy as np
import torch

from fgsim.config import conf
from fgsim.plot.xyscatter import binbourders_wo_outliers


def turnoff(step, decay):
    if decay == 0:
        return 1
    step *= decay
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
                np.diag(conf.training.smoothing.vars)
                * turnoff(step, conf.training.smoothing.decay),
                size=x.shape[:-1],
            )
        )
        .to(x.device)
        .float()
    )
    # for idx, name in enumerate(conf.loader.cell_prop_keys):
    #     plotdist(name, x[:, idx], presmooth=False)
    return x


def plotdist(name, arr, presmooth=True):
    dists = (
        torch.cdist(arr[:5000].reshape(1, -1, 1), arr[:5000].reshape(1, -1, 1))
        .reshape(-1)
        .cpu()
        .numpy()
    )
    fig, ax = plt.subplots()
    ax.hist(dists, bins=binbourders_wo_outliers(dists), histtype="bar")
    fig.savefig(
        f"wd/{name}-dists.png" if presmooth else f"wd/{name}-dists_smoothed.png"
    )
    plt.close()
