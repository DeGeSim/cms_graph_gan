from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

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
) -> plt.Figure:
    sim = to_np(sim)
    gen = to_np(gen)

    xrange, yrange = simranges(sim)

    sim_df = pd.DataFrame(
        {
            v1name: sim[:, 0],
            v2name: sim[:, 1],
            "cls": f"MC",
        }
    )
    gen_df = pd.DataFrame(
        {
            v1name: gen[:, 0],
            v2name: gen[:, 1],
            "cls": f"GAN",
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
        labels=["GAN", "MC"],
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
) -> plt.Figure:
    if len(sim) > 5000:
        sampleidxs = np.random.choice(sim.shape[0], size=5000, replace=False)
        sim = sim[sampleidxs]
        gen = gen[sampleidxs]

    xrange, yrange = simranges(sim)

    sim_df = pd.DataFrame(
        {
            v1name: sim[:, 0],
            v2name: sim[:, 1],
            "cls": f"MC",
        }
    )
    gen_df = pd.DataFrame(
        {
            v1name: gen[:, 0],
            v2name: gen[:, 1],
            "cls": f"GAN",
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
        labels=["GAN", "MC"],
    )

    return g.figure


def chip_to_binborders(arr, binborders):
    return np.clip(arr, binborders[0], binborders[-1])


@wrap_torch_to_np
def xy_hist(
    sim: np.ndarray,
    gen: np.ndarray,
    title: str,
    v1name: str,
    v2name: str,
    step: Optional[int] = None,
) -> plt.Figure:
    plt.cla()
    plt.clf()

    sns.set()
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    xedges = binbourders_wo_outliers(sim[:, 0])
    yedges = binbourders_wo_outliers(sim[:, 1])
    axes[0].hist2d(
        chip_to_binborders(sim[:, 0], xedges),
        chip_to_binborders(sim[:, 1], yedges),
        bins=[xedges, yedges],
    )

    axes[1].hist2d(
        chip_to_binborders(gen[:, 0], xedges),
        chip_to_binborders(gen[:, 1], yedges),
        bins=[xedges, yedges],
    )
    axes[0].set_title("MC")
    axes[1].set_title("GAN")
    axes[0].set(xlabel=v1name, ylabel=v2name)
    axes[1].set(xlabel=v1name, ylabel=v2name)

    if step is not None:
        title += f"\nStep {step}"
    fig.suptitle(title)
    return fig


@wrap_torch_to_np
def simranges(sim: np.ndarray):
    xrange = bounds_wo_outliers(sim[:, 0])
    yrange = bounds_wo_outliers(sim[:, 1])
    return xrange, yrange


@wrap_torch_to_np
def binbourders_wo_outliers(points: np.ndarray, bins=50) -> np.ndarray:
    return np.linspace(*bounds_wo_outliers(points), num=bins, endpoint=True)


@wrap_torch_to_np
def bounds_wo_outliers(points: np.ndarray) -> tuple:
    median = np.median(points, axis=0)

    # med_abs_lfluk = np.sqrt(np.mean((points[points < median] - median) ** 2))
    # med_abs_ufluk = np.sqrt(np.mean((points[points > median] - median) ** 2))
    # upper = median + max(med_abs_ufluk,med_abs_ufluk)
    # lower = median - max(med_abs_ufluk,med_abs_ufluk)
    outlier_scale = (
        max(
            np.abs(np.quantile(points, 0.99) - median),
            np.abs(np.quantile(points, 0.1) - median),
        )
        * 1.1
    )
    upper = median + outlier_scale
    lower = median - outlier_scale
    # print(lower,np.min(points), upper,np.max(points))
    upper = np.min([upper, np.max(points)])
    lower = np.max([lower, np.min(points)])
    return lower, upper
