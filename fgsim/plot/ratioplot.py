from typing import Optional

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import ScalarFormatter

from .binborders import binborders_wo_outliers, bincenters
from .infolut import var_to_bins, var_to_label

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    color=[
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#984ea3",
        "#f781bf",
        "#a65628",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]
)


def simlabel():
    from fgsim.config import conf

    if conf.dataset_name == "jetnet":
        if conf.loader.n_points == 150:
            return "\\JNl"
        if conf.loader.n_points == 30:
            return "\\JNs"
    elif conf.dataset_name == "calochallange":
        return "CC Dataset 2"
    else:
        return "Simulation"


def ratioplot(
    arrays: list[np.ndarray],
    ftn: str,
    labels: Optional[list[str]] = None,  # Labels for the datasets
    weights: Optional[list[Optional[np.ndarray]]] = None,
    bins: Optional[np.ndarray] = None,
) -> Figure:
    plt.close("all")
    # set defaults
    if labels is None:
        if len(arrays) == 2:
            labels = [simlabel(), "\\dt"]
        else:
            raise Exception("Need to specify array labels.")

    if bins is None:
        bins = var_to_bins(ftn)
        if bins is None:
            bins = binborders_wo_outliers(arrays[0])

    weighted = weights[0] is not None
    if not weighted:
        weights = [None for _ in arrays]

    title = var_to_label(ftn)

    # check lens
    assert len(arrays) == len(labels) == len(weights)
    len(arrays)

    arrays = [
        e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e
        for e in arrays
    ]
    if weighted:
        weights = [e.detach().cpu().numpy() for e in weights]
        # use the simulation as a reference to scale the sum of the weights to 1
        simulation_factor = weights[0].shape[0] / weights[0].sum()
        for iarr in range(len(weights)):
            weights[iarr] = weights[iarr] * simulation_factor

    hists = [
        np.histogram(arr, bins=bins, weights=w)[0]
        for arr, w in zip(arrays, weights)
    ]

    if weighted:
        _weights = [e / np.sum(e) for e in weights]
        errors = [
            np.sqrt(np.histogram(arr, bins=bins, weights=w**2)[0])
            for arr, w in zip(arrays, _weights)
        ]
        del _weights
    else:
        errors = [np.sqrt(hist) for hist in hists]
        pass

    sim_hist = hists[0]

    ax: Axes
    axrat: Axes
    fig, (ax, axrat) = plt.subplots(
        2,
        1,
        figsize=(5.5, 6),
        gridspec_kw={"height_ratios": [2, 0.7]},
        sharex="col",
    )

    colors = make_top_hists(hists, arrays, bins, labels, errors, ax)

    axrat.set_ylim(0.48, 1.52)

    #### errors and out-of-range markers ###
    x = bincenters(bins)
    frac_error_x = np.array([(bins[1] - bins[0]) / 2.0] * (len(bins) - 1))
    n_oor_upper = np.ones_like(x)
    n_oor_lower = np.ones_like(x)
    for ihist, (gen_hist, gen_error, label, color) in enumerate(
        zip(hists, errors, labels, colors)
    ):
        ratio, high, low, nan_mask = ratio_errors(
            gen_hist, sim_hist, gen_error, errors[0]
        )

        # simulation
        if ihist == 0:
            # grey r==1 line
            axrat.axhline(1, color=color)
            axrat.fill_between(
                x[nan_mask],
                ratio[nan_mask] - low[nan_mask],
                ratio[nan_mask] + high[nan_mask],
                color=color,
                label=label,
                alpha=0.4,
            )
        # models
        else:
            axrat.errorbar(
                x=x[nan_mask],
                y=ratio[nan_mask],
                yerr=np.stack([low[nan_mask], high[nan_mask]]),
                xerr=frac_error_x[nan_mask],
                barsabove=True,
                linestyle="",
                marker=None,
                ecolor=color,  # Use different color for each dataset
                label=label,  # Add label
                markersize=2,
            )
            n_oor_upper, n_oor_lower = make_oor_indicators(
                axrat, x, ratio, bins, color, n_oor_upper, n_oor_lower
            )

    if not weighted:
        ax.set_ylabel("Counts per Bin", fontsize=17)
    else:
        ax.set_ylabel("Sum of Weights per Bin", fontsize=17)
    ax.legend(fontsize=14, loc="best", framealpha=1, edgecolor="black")

    if len(arrays) > 2:
        axrat.set_ylabel(r"$\frac{\text{Dataset}}{\text{Simulation}}$", fontsize=15)
    else:
        axrat.set_ylabel(r"$\frac{\text{Model}}{\text{Simulation}}$", fontsize=15)

    # horizontal grid
    ax.grid(True, "major", "y")

    # bounding box around the plots
    for iax in [ax, axrat]:
        for spline in iax.spines.values():
            spline.set_linewidth(1)
            spline.set_color("black")
    axrat.set_xlabel(title, fontsize=26)

    ax.tick_params(axis="y", which="both", labelsize=15)
    axrat.tick_params(axis="y", which="both", labelsize=15)

    ## ## x Ticks in the middle:
    # needed to access the formatter, run mpl logic
    # to reduce the number of ticks if there are too many
    plt.tight_layout()
    plt.draw()

    # remove labels
    ax.tick_params("x", labelbottom=False)
    axrat.tick_params("x", labelbottom=False)
    # safe for later
    xtickpos = ax.xaxis.get_ticklocs()[1:-1]
    xtickformatter = ax.xaxis.get_major_formatter()
    xtickformatter.format = xtickformatter.format.replace("%1.3f", "%1.3g")
    xticklabels = [xtickformatter(e) for e in xtickpos]

    # ticks to top for ratio Axes
    axrat.xaxis.tick_top()

    # Calculate the middle position for the shared x-tick labels
    # It's the average of the top of the bottom subplot
    # and the bottom of the top subplot
    middle = (ax.get_position().ymin + axrat.get_position().ymax) / 2
    # Add shared x-tick labels as text annotations
    for xpos, label in zip(xtickpos, xticklabels):
        display_coord = ax.transData.transform((xpos, 0))
        fig_coord = fig.transFigure.inverted().transform(display_coord)
        plt.figtext(
            fig_coord[0],
            middle,
            label,
            ha="center",
            va="center",
            fontsize=14,
        )
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)  # Adjust the spacing if needed
    fig.savefig("/home/mscham/fgsim/wd/tmp.pdf")
    print("Done")
    return fig


def make_top_hists(hists, arrays, bins, labels, errors, ax):
    ### run histogramms ###
    artists = mplhep.histplot(
        np.stack(hists),
        bins=bins,
        label=labels,  # Use custom labels
        yerr=np.stack(errors),
        ax=ax,
    )

    #### overflow bins ###
    delta = (bins[1] - bins[0]) / 2
    colors = [a.stairs.get_edgecolor() for a in artists]
    kwstyle = dict(
        # linestyle=(0, (0.5, 0.3)),
        lw=3,
        path_effects=[
            # path_effects.Stroke(linewidth=5, foreground="black"),
            path_effects.PathPatchEffect(
                edgecolor="black",
                linewidth=5,
            ),
            path_effects.Normal(),
        ],
    )
    ibar = 1
    for arr, color in zip(arrays, colors):
        undershot = (arr < bins[0]).sum()
        if undershot > 0:
            ax.vlines(
                x=bins[0] - ibar * delta * kwstyle["lw"],
                ymin=0,
                ymax=undershot,
                color=color,
                **kwstyle,
            )
            ibar += 1
    ibar = 1
    for arr, color in zip(arrays, colors):
        overshot = (arr > bins[-1]).sum()
        if overshot > 0:
            ax.vlines(
                x=bins[-1] + ibar * delta * kwstyle["lw"],
                ymin=0,
                ymax=overshot,
                color=color,
                **kwstyle,
            )
            ibar += 1
    sim_hist = hists[0]
    if (sim_hist > (sim_hist.max() / 10)).mean() < 0.1:
        ax.set_yscale("log")
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(13)
    return colors


def make_oor_indicators(axrat, x, ratio, bins, color, n_oor_upper, n_oor_lower):
    # Indicators for out of range ratios:
    lower, upper = axrat.get_ylim()
    ms = np.diff(
        axrat.transData.transform(np.stack([bins, np.zeros_like(bins)]).T)[:, 0]
    )

    tup = np.stack([np.zeros_like(x), np.ones_like(x)]).T
    tup = axrat.transAxes.transform(tup)
    tup[:, 1] -= (ms + 10 / ms) * n_oor_upper
    tup = axrat.transData.inverted().transform(tup)
    yupper = tup[:, 1]

    tup = np.stack([np.zeros_like(x), np.zeros_like(x)]).T
    tup = axrat.transAxes.transform(tup)
    tup[:, 1] += (ms + 10 / ms) * n_oor_lower
    tup = axrat.transData.inverted().transform(tup)
    ylower = tup[:, 1]

    oor_idxs = np.where(ratio > upper)[0]
    n_oor_upper[oor_idxs] += 1
    axrat.scatter(
        x=x[oor_idxs],
        y=yupper[oor_idxs],
        marker="^",
        color=color,
        s=ms[oor_idxs],
    )
    oor_idxs = np.where(ratio < lower)[0]
    n_oor_lower[oor_idxs] += 1
    axrat.scatter(
        x=x[oor_idxs],
        y=ylower[oor_idxs],
        marker="v",
        color=color,
        s=ms[oor_idxs],
    )
    return n_oor_upper, n_oor_lower


def ratio_errors(gen_hist, sim_hist, gen_error, sim_error):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = gen_hist / sim_hist
        ratio_error_y = np.abs(ratio) * np.sqrt(
            (sim_error / sim_hist) ** 2 + (gen_error / gen_hist) ** 2
        )
        nan_mask = (ratio != 0) & np.invert(np.isnan(ratio_error_y))
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     nan_mask = sim_hist != 0
    # ratio = gen_hist / sim_hist

    # level = 0.682689492137
    # alpha = 1 - level
    # k = gen_hist
    # n = gen_hist + sim_hist
    # low, high = beta.ppf([alpha / 2, 1 - alpha / 2], [k, k + 1], [n - k + 1, n - k])

    # eff = k / n
    # ratio = eff / (1 - eff)
    # low = low / (1 - low)
    # high = high / (1 - high)
    return ratio, ratio_error_y, ratio_error_y, nan_mask
    # return ratio, high, low, nan_mask
