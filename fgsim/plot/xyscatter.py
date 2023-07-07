from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from fgsim.plot.binborders import (
    binborders_wo_outliers,
    bounds_wo_outliers,
    chip_to_binborders,
)
from fgsim.utils.torchtonp import wrap_torch_to_np

np.set_printoptions(formatter={"float_kind": "{:.3g}".format})


def to_np(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.clone().detach().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError


@wrap_torch_to_np
def gausstr(sim: np.ndarray, gen: np.ndarray):
    mean_sim = np.around(np.mean(sim, axis=0), 2)
    cov_sim = str(np.around(np.cov(sim, rowvar=0), 2)).replace("\n", "")
    mean_gen = np.around(np.mean(gen, axis=0), 2)
    cov_gen = str(np.around(np.cov(gen, rowvar=0), 2)).replace("\n", "")
    return [f"GAN μ{mean_gen}\nσ{cov_gen}", f"MC μ{mean_sim}\nσ{cov_sim}"]


def xyscatter(
    sim: Union[np.ndarray, torch.Tensor],
    gen: Union[np.ndarray, torch.Tensor],
    title: str,
    v1name: str,
    v2name: str,
) -> Figure:
    sim = to_np(sim)
    gen = to_np(gen)

    xrange, yrange = simranges(sim)

    sim_df = pd.DataFrame(
        {
            v1name: sim[:, 0],
            v2name: sim[:, 1],
            "cls": "MC",
        }
    )
    gen_df = pd.DataFrame(
        {
            v1name: gen[:, 0],
            v2name: gen[:, 1],
            "cls": "DeepTreeGAN",
        }
    )
    df = pd.concat([sim_df, gen_df], ignore_index=True)

    plt.cla()
    plt.clf()
    g: sns.JointGrid = sns.jointplot(
        data=df,
        x=v1name,
        y=v2name,
        hue="cls",
        legend=False,
        xlim=xrange,
        ylim=yrange,
    )
    g.fig.suptitle(title)

    g.figure.subplots_adjust(top=0.95)
    plt.legend(
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        labels=["DeepTreeGAN", "MC"],
    )

    return g.figure


@wrap_torch_to_np
def xyscatter_faint(
    sim: np.ndarray,
    gen: np.ndarray,
    title: str,
    v1name: str,
    v2name: str,
    step: Optional[int] = None,
) -> Figure:
    if len(sim) > 5000:
        sampleidxs = np.random.choice(sim.shape[0], size=5000, replace=False)
        sim = sim[sampleidxs]
        gen = gen[sampleidxs]

    xrange, yrange = simranges(sim)

    sim_df = pd.DataFrame(
        {
            v1name: sim[:, 0],
            v2name: sim[:, 1],
            "cls": "MC",
        }
    )
    gen_df = pd.DataFrame(
        {
            v1name: gen[:, 0],
            v2name: gen[:, 1],
            "cls": "DeepTreeGAN",
        }
    )
    df = pd.concat([sim_df, gen_df], ignore_index=True)

    plt.cla()
    plt.clf()
    g: sns.JointGrid = sns.jointplot(
        data=df,
        x=v1name,
        y=v2name,
        hue="cls",
        alpha=0.15,
        legend=False,
        xlim=xrange,
        ylim=yrange,
    )
    if step is not None:
        title += f"\nStep {step}"
    g.fig.suptitle(title)
    # g.ax_joint.collections[0].set_alpha(0)
    # g.fig.tight_layout()
    g.figure.subplots_adjust(top=0.95)
    plt.legend(
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        labels=["DeepTreeGAN", "MC"],
    )

    return g.figure


@wrap_torch_to_np
def xy_hist(
    sim: np.ndarray,
    gen: np.ndarray,
    title: str,
    v1name: str,
    v2name: str,
    v1bins: Optional[np.ndarray] = None,
    v2bins: Optional[np.ndarray] = None,
    step: Optional[int] = None,
) -> Figure:
    plt.cla()
    plt.clf()

    sns.set()
    fig: Figure
    sim_axes: Axes
    gen_axes: Axes
    fig, (sim_axes, gen_axes) = plt.subplots(
        1, 2, sharex=True, sharey=True  # , figsize=(20, 12)
    )
    if v1bins is None:
        xedges = binborders_wo_outliers(sim[:, 0])
        yedges = binborders_wo_outliers(sim[:, 1])
    else:
        xedges = v1bins
        yedges = v2bins

    # flip x and y axis if the x val has more bins
    if len(xedges) > len(yedges) + 5:
        xedges, yedges = yedges, xedges
        sim = sim[:, (1, 0)]
        gen = gen[:, (1, 0)]
        v1bins, v2bins = v2bins, v1bins
        v1name, v2name = v2name, v1name

    shist, _, _ = np.histogram2d(
        chip_to_binborders(sim[:, 0], xedges),
        chip_to_binborders(sim[:, 1], yedges),
        bins=(xedges, yedges),
    )  # , density=density, weights=weights)
    ghist, _, _ = np.histogram2d(
        chip_to_binborders(gen[:, 0], xedges),
        chip_to_binborders(gen[:, 1], yedges),
        bins=(xedges, yedges),
    )

    if (shist > (shist.max() / 10)).mean() < 0.1:
        norm = LogNorm(max(shist.min(), 1), shist.max())
    else:
        norm = Normalize(shist.min(), shist.max())

    for iax, (ax, hist) in enumerate(zip([sim_axes, gen_axes], [shist, ghist])):
        im = ax.imshow(hist.T, cmap=plt.cm.coolwarm, norm=norm)  # , aspect="equal")
        if iax == 0:
            ax.set_ylabel(v2name)
            ax.set_yticks(**edge_to_labels(yedges))
        ax.set_xticks(rotation=45, **edge_to_labels(xedges))
        ax.set_xlabel(v1name)

    # s_cax = sim_axes.inset_axes([1.04, 0.2, 0.05, 0.6])
    fig.colorbar(im)

    sim_axes.set_title("MC")
    gen_axes.set_title("DeepTreeGAN")

    if step is not None:
        title += f"\nStep {step}"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


@wrap_torch_to_np
def simranges(sim: np.ndarray):
    xrange = bounds_wo_outliers(sim[:, 0])
    yrange = bounds_wo_outliers(sim[:, 1])
    return xrange, yrange


def edge_to_labels(edges):
    if len(edges) < 10:
        ticks = np.arange(len(edges))
    else:
        ticks = np.linspace(0, len(edges) - 1, 10, dtype="int")
    labels = [str(int(e)) for e in edges[ticks]]
    return dict(ticks=ticks, labels=labels)
