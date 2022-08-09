from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

np.set_printoptions(formatter={"float_kind": "{:.3g}".format})


def to_np(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.clone().detach().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError


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


def xyscatter_faint(
    sim: np.ndarray,
    gen: np.ndarray,
    title: str,
    v1name: str,
    v2name: str,
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


def xy_hist(
    sim: np.ndarray,
    gen: np.ndarray,
    title: str,
    v1name: str,
    v2name: str,
) -> plt.Figure:
    plt.cla()
    plt.clf()

    sns.set()
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    _, xedges, yedges, _ = axes[0].hist2d(
        sim[:, 0],
        sim[:, 1],
        bins=[
            binbourders_wo_outliers(sim[:, 0]),
            binbourders_wo_outliers(sim[:, 1]),
        ],
    )

    axes[1].hist2d(
        gen[:, 0],
        gen[:, 1],
        bins=[xedges, yedges],
    )
    axes[0].set_title("MC")
    axes[1].set_title("GAN")
    axes[0].set(xlabel=v1name, ylabel=v2name)
    axes[1].set(xlabel=v1name, ylabel=v2name)

    fig.suptitle(title)
    return fig


def simranges(sim: np.ndarray):
    xrange = bounds_wo_outliers(sim[:, 0])
    yrange = bounds_wo_outliers(sim[:, 1])
    return xrange, yrange


def binbourders_wo_outliers(points) -> np.ndarray:
    return np.linspace(*bounds_wo_outliers(points), num=100, endpoint=True)


def bounds_wo_outliers(points: np.ndarray) -> tuple:
    thresh = 3.0
    median = np.median(points, axis=0)
    med_abs_lfluk = np.sqrt(np.median((points[points < median] - median) ** 2))
    med_abs_ufluk = np.sqrt(np.median((points[points > median] - median) ** 2))
    upper = median + med_abs_ufluk * thresh
    lower = median - med_abs_lfluk * thresh
    # print(lower,np.min(points), upper,np.max(points))
    upper = np.min([upper, np.max(points)])
    lower = np.max([lower, np.min(points)])
    return lower, upper
