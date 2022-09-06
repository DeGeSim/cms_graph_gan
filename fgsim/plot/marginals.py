from typing import Dict

import matplotlib.pyplot as plt
import mplhep
import numpy as np
from matplotlib.figure import Figure

from fgsim.config import conf

from .xyscatter import binbourders_wo_outliers


def ftx_marginals(
    sim,
    gen,
) -> Dict[str, Figure]:
    sim_features = {
        varname: arr
        for varname, arr in zip(
            conf.loader.cell_prop_keys,
            sim.x.reshape(-1, conf.loader.n_features).T.cpu().numpy(),
        )
    }
    gen_features = {
        varname: arr
        for varname, arr in zip(
            conf.loader.cell_prop_keys,
            gen.x.reshape(-1, conf.loader.n_features).T.cpu().numpy(),
        )
    }
    plots_d: Dict[str, Figure] = {}
    for ftn in conf.loader.cell_prop_keys:
        fig, (ax, axrat) = plt.subplots(
            2,
            1,
            figsize=(6, 8),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        sim_arr = sim_features[ftn]
        gen_arr = gen_features[ftn]

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
        ax.set_title(ftn)
        axrat.set_ylim(0, 2)
        axrat.set_xticks([])
        axrat.set_xticklabels([])
        plots_d[f"ftxmarginal_{ftn}.pdf"] = fig

    return plots_d
