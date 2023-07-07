from typing import Optional

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from matplotlib.figure import Figure

from fgsim.utils.torchtonp import wrap_torch_to_np

from .binborders import binborders_wo_outliers


@wrap_torch_to_np
def ratioplot(
    sim: np.ndarray, gen: np.ndarray, title: str, bins: Optional[np.ndarray] = None
) -> Figure:
    fig, (ax, axrat) = plt.subplots(
        2,
        1,
        figsize=(6, 7),
        gridspec_kw={"height_ratios": [2, 0.7]},
    )
    # if title == "Disc Score":
    #     bins = binborders_by_bounds(0, 1)
    # else:
    if bins is None:
        bins = binborders_wo_outliers(sim)
    n_bins = len(bins) - 1

    sim_hist, _ = np.histogram(sim, bins=bins)
    gen_hist, _ = np.histogram(gen, bins=bins)
    sim_error = np.sqrt(sim_hist)
    gen_error = np.sqrt(gen_hist)

    mplhep.histplot(
        [sim_hist, gen_hist],
        bins=bins,
        label=["Simulation", "DeepTree"],
        yerr=[sim_error, gen_error],
        ax=ax,
    )
    # overflow bins
    delta = (bins[1] - bins[0]) / 2
    simcolor = ax.containers[1][0]._color
    gencolor = ax.containers[2][0]._color
    kwstyle = dict(linestyle=(0, (0.5, 0.3)), lw=3)
    for arr, color, factor in zip([gen, sim], [gencolor, simcolor], [1, 2]):
        ax.vlines(
            x=bins[0] - factor * delta,
            ymin=0,
            ymax=(arr < bins[0]).sum(),
            color=color,
            **kwstyle,
        )
        ax.vlines(
            x=bins[-1] + factor * delta,
            ymin=0,
            ymax=(arr > bins[-1]).sum(),
            color=color,
            **kwstyle,
        )

    if (sim_hist > (sim_hist.max() / 10)).mean() < 0.1:
        ax.set_yscale("log")
    ax.set_ylabel("Frequency", fontsize=16)
    # ax.get_yaxis().set_ticks([])

    ax.legend(fontsize=16, loc="upper left")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    # ratioplot
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = gen_hist / sim_hist
        frac_error_y = np.abs(frac) * np.sqrt(
            (sim_error / sim_hist) ** 2 + (gen_error / gen_hist) ** 2
        )
        # frac_error_x = np.array([(bins[1] - bins[0]) / 2.0] * n_bins)
        frac_mask = (frac != 0) & np.invert(np.isnan(frac_error_y))
    axrat.axhline(1, color="grey")

    axrat.errorbar(
        x=np.array(range(n_bins))[frac_mask],
        y=frac[frac_mask],
        yerr=frac_error_y[frac_mask],
        xerr=0.5,  # frac_error_x[frac_mask],
        barsabove=True,
        linestyle="",
        marker="o",
        ecolor="black",
        markersize=2,
    )
    # axrat.plot(frac, marker="o", linestyle="", markersize=4)

    axrat.set_ylim(0.5, 1.5)
    axrat.set_xticks([])
    axrat.set_xticklabels([])

    fig.suptitle(title, fontsize=30)
    plt.tight_layout()
    return fig
