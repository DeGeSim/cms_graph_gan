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
    # sim_features = jetnet.utils.jet_features(sim)
    # gen_features = jetnet.utils.jet_features(gen)
    sim_features = {
        varname: arr for varname, arr in zip(conf.loader.cell_prop_keys, sim.T)
    }
    gen_features = {
        varname: arr for varname, arr in zip(conf.loader.cell_prop_keys, gen.T)
    }
    ft_names = list(sim_features.keys())
    plt.cla()
    plt.clf()
    sns.set()
    fig, axes = plt.subplots(
        2, len(ft_names), figsize=(18, 8), gridspec_kw={"height_ratios": [2, 1]}
    )
    for (ax, axaux), ftn in zip(zip(*axes), ft_names):
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

        ax.set_title({"mass": "Mass [GeV]", "pt": "$p_T$ [GeV]", "eta": "η"}[ftn])
        ax.set_yticks([])
        ax.set_yticklabels([])
        if ax is axes[0]:
            ax.set_ylabel("Frequency")
        if ax is axes[-1]:
            ax.legend(["MC", "GAN"])
        frac = gen_hist / sim_hist
        axaux.plot(frac)
        axaux.set_ylim(0, 2)
        axaux.axhline(1, color="black")
    fig.suptitle("Jet features")
    plt.tight_layout()
    return fig
