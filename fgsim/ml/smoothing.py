import matplotlib.pyplot as plt
import numpy as np
import torch

from fgsim.config import conf
from fgsim.io.sel_loader import scaler
from fgsim.plot.xyscatter import binbourders_wo_outliers

# This code scales the smearning standard deviations
var_names = conf.loader.x_features
n_vars = len(var_names)
scaled_stds = np.array([0.0] * n_vars)
for ivar in range(n_vars):
    std_unscaled = conf.training.smoothing.stds[ivar]
    if std_unscaled == 0:
        continue
    iscaler = scaler.transfs[ivar]
    zero_tf = iscaler.transform(torch.tensor(0.0).reshape(1, -1))
    scaled_stds[ivar] = (
        iscaler.transform(np.array(std_unscaled).reshape(1, -1)) - zero_tf
    )


def turnoff(step, decay):
    if decay == 0:
        return 1
    step *= decay
    if step > np.pi:
        return 0
    else:
        return (np.cos(step) + 1) / 2


def smooth_features(x, step):
    # x_inv_transf = scaler.inverse_transform(x)
    # for idx, name in enumerate(var_names):
    #     plotdist(name, x_inv_transf[:, idx])

    x = (
        x
        + torch.from_numpy(
            np.random.multivariate_normal(
                [0.0] * conf.loader.n_features,
                np.diag(
                    (scaled_stds * turnoff(step, conf.training.smoothing.decay))
                    ** 2
                ),
                size=x.shape[:-1],
            )
        )
        .to(x.device)
        .float()
    )
    # x_inv_transf = scaler.inverse_transform(x)
    # for idx, (name, std) in enumerate(zip(var_names, conf.training.smoothing.stds)):
    #     plotdist(name, x_inv_transf[:, idx], std=std, presmooth=False)
    return x


def plotdist(name, arr, std=None, presmooth=True):
    dists = (
        torch.cdist(arr[:5000].reshape(1, -1, 1), arr[:5000].reshape(1, -1, 1))
        .reshape(-1)
        .cpu()
        .numpy()
    )
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.hist(dists, bins=binbourders_wo_outliers(dists, bins=300), histtype="bar")
    ax1.set_title("cdist histogram")
    ax2.hist(
        arr[:5000],
        bins=binbourders_wo_outliers(arr[:5000], bins=300),
        histtype="bar",
    )
    ax2.set_title("marginal histogram")

    fig.suptitle(
        f"Hists for {name} w/o smoothing"
        if presmooth
        else f"Hists for {name} w/ std {std}"
    )
    fig.tight_layout()
    fig.savefig(
        f"wd/{name}-dists.png"
        if presmooth
        else f"wd/{name}-{std}-dists_smoothed.png"
    )
    plt.close()
