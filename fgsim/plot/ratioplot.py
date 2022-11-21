from typing import Optional

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from matplotlib.figure import Figure

from fgsim.utils.torchtonp import wrap_torch_to_np

from .xyscatter import binbourders_wo_outliers


@wrap_torch_to_np
def ratioplot(
    sim: np.ndarray,
    gen: np.ndarray,
    title: str,
    step: Optional[int] = None,
) -> Figure:
    fig, (ax, axrat) = plt.subplots(
        2,
        1,
        figsize=(6, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    bins = binbourders_wo_outliers(sim)
    n_bins = len(bins) - 1

    sim_hist, sim_bins = np.histogram(sim, bins=bins)
    gen_hist, _ = np.histogram(gen, bins=bins)
    sim_error = np.sqrt(sim_hist)
    gen_error = np.sqrt(gen_hist)

    mplhep.histplot(
        [sim_hist, gen_hist],
        bins=sim_bins,
        label=["MC", "GAN"],
        yerr=[sim_error, gen_error],
        ax=ax,
    )

    ax.set_ylabel("Frequency")
    ax.legend()

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

    axrat.set_ylim(0, 2)
    axrat.set_xticks([])
    axrat.set_xticklabels([])

    if step is not None:
        title += f"\nStep {step}"
    fig.suptitle(title)
    plt.tight_layout()
    return fig
