import jetnet
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import seaborn as sns

from fgsim.config import conf

from .xyscatter import binbourders_wo_outliers


def jet_features(
    sim: np.ndarray,
    gen: np.ndarray,
) -> plt.Figure:
    sim_features_agr = jetnet.utils.jet_features(sim)
    gen_features_agr = jetnet.utils.jet_features(gen)

    plt.cla()
    plt.clf()
    sns.set()
    fig, axes = plt.subplots(
        4,
        3,
        figsize=(18, 14),
        gridspec_kw={"height_ratios": [2, 1, 2, 1]},
    )
    for (ax, axrat), ftn in zip(zip(*axes[:2]), ["pt", "eta", "mass"]):
        sim_arr = sim_features_agr[ftn]
        gen_arr = gen_features_agr[ftn]

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

        ax.set_title(
            {
                "mass": "$m_{rel}$",
                "phi": "$Σ ϕ_{rel}$",
                "pt": "$Σp_{T,rel}$",
                "eta": "$Ση_{rel}$",
            }[ftn]
        )
        if ax is axes[0][0]:
            ax.set_ylabel("Frequency")
        if ax is axes[0][-1]:
            ax.legend(["MC", "GAN"])
        frac = gen_hist / sim_hist
        axrat.plot(frac)
        axrat.set_ylim(0, 2)
        axrat.axhline(1, color="black")
        axrat.set_xticks([])
        axrat.set_xticklabels([])

    sim_features = {
        varname: arr
        for varname, arr in zip(conf.loader.cell_prop_keys, sim.reshape(-1, 3).T)
    }
    gen_features = {
        varname: arr
        for varname, arr in zip(conf.loader.cell_prop_keys, gen.reshape(-1, 3).T)
    }

    for (ax, axrat), ftn in zip(zip(*axes[2:4]), ["pt", "eta", "phi"]):
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

        if ax is axes[1][0]:
            ax.set_ylabel("Frequency")

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

    fig.suptitle("Jet features")
    plt.tight_layout()
    return fig
