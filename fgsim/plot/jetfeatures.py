from typing import Dict

import jetnet
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from fgsim.config import conf
from fgsim.utils.jetnetutils import to_stacked_mask

from .xyscatter import binbourders_wo_outliers


def jet_features(
    sim,
    gen,
) -> Dict[str, Figure]:
    sim_features_agr = jetnet.utils.jet_features(
        to_stacked_mask(sim).cpu().numpy()[..., :3]
    )
    gen_features_agr = jetnet.utils.jet_features(
        to_stacked_mask(gen).cpu().numpy()[..., :3]
    )

    plt.cla()
    plt.clf()
    sns.set()
    feature_name_d = {
        "mass": "$m_{rel}$",
        "phi": "$Σ ϕ_{rel}$",
        "pt": "$Σp_{T,rel}$",
        "eta": "$Ση_{rel}$",
    }
    plots_d = {}

    for ftn in ["pt", "eta", "mass"]:
        fig, (ax, axrat) = plt.subplots(
            2,
            1,
            figsize=(6, 8),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        sim_arr = sim_features_agr[ftn]
        gen_arr = gen_features_agr[ftn]

        # upper plot
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

        ax.set_title(feature_name_d[ftn])

        ax.set_ylabel("Frequency")
        ax.legend(["MC", "GAN"])
        # ratio plot
        frac = gen_hist / sim_hist
        axrat.plot(frac)
        axrat.set_ylim(0, 2)
        axrat.axhline(1, color="black")
        axrat.set_xticks([])
        axrat.set_xticklabels([])
        plt.tight_layout()
        plots_d[f"jetfeatures_{ftn}.pdf"] = fig

    sim_features = {
        varname: arr
        for varname, arr in zip(
            conf.loader.x_features, sim.x.reshape(-1, 3).T.cpu().numpy()
        )
    }
    gen_features = {
        varname: arr
        for varname, arr in zip(
            conf.loader.x_features, gen.x.reshape(-1, 3).T.cpu().numpy()
        )
    }

    for ftn in ["pt", "eta", "phi"]:
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
        frac = gen_hist / sim_hist
        axrat.plot(frac)
        axrat.axhline(1, color="black")
        ax.set_title(
            {
                "pt": "$p_T$ [GeV]",
                "eta": "$η_{rel}$",
                "phi": "$ϕ_{rel}$",
            }[ftn]
        )
        axrat.set_ylim(0, 2)
        axrat.set_xticks([])
        axrat.set_xticklabels([])
        plots_d[f"pfeatures_{ftn}.pdf"] = fig

    return plots_d
