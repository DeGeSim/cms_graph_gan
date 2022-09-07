import matplotlib.pyplot as plt
import mplhep
import numpy as np
from matplotlib.figure import Figure

from .xyscatter import binbourders_wo_outliers


def ratioplot(sim_arr, gen_arr, title) -> Figure:
    fig, (ax, axrat) = plt.subplots(
        2,
        1,
        figsize=(6, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    bins = binbourders_wo_outliers(sim_arr)

    sim_hist, sim_bins = np.histogram(sim_arr, bins=bins)
    gen_hist, _ = np.histogram(gen_arr, bins=bins)
    mplhep.histplot(
        [sim_hist, gen_hist],
        bins=sim_bins,
        label=["MC", "GAN"],
        yerr=[np.sqrt(sim_hist), np.sqrt(gen_hist)],
        ax=ax,
    )

    ax.set_ylabel("Frequency")

    # ratioplot
    with np.errstate(divide="ignore"):
        frac = gen_hist / sim_hist
    axrat.plot(frac)
    axrat.axhline(1, color="black")
    axrat.set_ylim(0, 2)
    axrat.set_xticks([])
    axrat.set_xticklabels([])
    fig.suptitle(title)
    return fig
