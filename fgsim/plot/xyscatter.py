from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def to_np(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.clone().detach().cpu().numpy()
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise TypeError


def xyscatter(
    sim: Union[np.ndarray, torch.Tensor],
    gen: Union[np.ndarray, torch.Tensor],
    title: str,
) -> plt.Figure:
    sim = to_np(sim)
    gen = to_np(gen)
    np.set_printoptions(formatter={"float_kind": "{:.3g}".format})
    mean_sim = np.around(np.mean(sim, axis=0), 2)
    cov_sim = str(np.around(np.cov(sim, rowvar=0), 2)).replace("\n", "")
    mean_gen = np.around(np.mean(gen, axis=0), 2)
    cov_gen = str(np.around(np.cov(gen, rowvar=0), 2)).replace("\n", "")
    sim_df = pd.DataFrame(
        {
            "x": sim[:, 0],
            "y": sim[:, 1],
            "cls": f"sim μ{mean_sim}\nσ{cov_sim}",
        }
    )
    gen_df = pd.DataFrame(
        {
            "x": gen[:, 0],
            "y": gen[:, 1],
            "cls": f"gen μ{mean_gen}\nσ{cov_gen}",
        }
    )
    df = pd.concat([sim_df, gen_df], ignore_index=True)

    plt.cla()
    plt.clf()
    g: sns.JointGrid = sns.jointplot(data=df, x="x", y="y", hue="cls", legend=False)
    g.fig.suptitle(title)
    g.figure
    # g.ax_joint.collections[0].set_alpha(0)
    # g.fig.tight_layout()
    g.figure.subplots_adjust(top=0.95)
    plt.legend(
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        labels=[f"gen μ{mean_gen}\nσ{cov_gen}", f"sim μ{mean_sim}\nσ{cov_sim}"],
    )

    return g.figure


def xyscatter_faint(sim: np.array, gen: np.array, title: str) -> plt.Figure:
    np.set_printoptions(formatter={"float_kind": "{:.3g}".format})
    mean_sim = np.around(np.mean(sim, axis=0), 2)
    cov_sim = str(np.around(np.cov(sim, rowvar=0), 2)).replace("\n", "")
    mean_gen = np.around(np.mean(gen, axis=0), 2)
    cov_gen = str(np.around(np.cov(gen, rowvar=0), 2)).replace("\n", "")
    sim_df = pd.DataFrame(
        {
            "x": sim[:, 0],
            "y": sim[:, 1],
            "cls": f"sim μ{mean_sim}\nσ{cov_sim}",
        }
    )
    gen_df = pd.DataFrame(
        {
            "x": gen[:, 0],
            "y": gen[:, 1],
            "cls": f"gen μ{mean_gen}\nσ{cov_gen}",
        }
    )
    df = pd.concat([sim_df, gen_df], ignore_index=True)

    plt.cla()
    plt.clf()
    g: sns.JointGrid = sns.jointplot(
        data=df, x="x", y="y", hue="cls", alpha=0.1, legend=False
    )
    g.fig.suptitle(title)
    # g.ax_joint.collections[0].set_alpha(0)
    # g.fig.tight_layout()
    g.figure.subplots_adjust(top=0.95)
    plt.legend(
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        labels=[f"gen μ{mean_gen}\nσ{cov_gen}", f"sim μ{mean_sim}\nσ{cov_sim}"],
    )

    return g.figure


def xy_hist(sim: np.array, gen: np.array, title: str) -> plt.Figure:
    np.set_printoptions(formatter={"float_kind": "{:.3g}".format})
    mean_sim = np.around(np.mean(sim, axis=0), 2)
    cov_sim = str(np.around(np.cov(sim, rowvar=0), 2)).replace("\n", "")
    mean_gen = np.around(np.mean(gen, axis=0), 2)
    cov_gen = str(np.around(np.cov(gen, rowvar=0), 2)).replace("\n", "")
    sim_df = pd.DataFrame(
        {
            "x": sim[:, 0],
            "y": sim[:, 1],
            "cls": f"sim μ{mean_sim}\nσ{cov_sim}",
        }
    )
    gen_df = pd.DataFrame(
        {
            "x": gen[:, 0],
            "y": gen[:, 1],
            "cls": f"gen μ{mean_gen}\nσ{cov_gen}",
        }
    )
    plt.cla()
    plt.clf()

    sns.set()
    fig, axes = plt.subplots(1, 2)
    axes[0].hist2d(sim_df["x"], sim_df["y"], bins=[100, 100])
    axes[0].set_title("sim")
    axes[1].hist2d(gen_df["x"], gen_df["y"], bins=[100, 100])
    axes[1].set_title("gen")
    fig.suptitle(title)
    return fig
