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
            labels = ["Simulation", "Model"]
        else:
            raise Exception("Need to specify array labels.")

    if bins is None:
        bins = var_to_bins(ftn)
        if bins is None:
            bins = binborders_wo_outliers(arrays[0])

    if weights is None:
        weights = [None for _ in arrays]

    title = var_to_label(ftn)

    # cehck lens
    assert len(arrays) == len(labels) == len(weights)
    len(arrays)

    arrays = [
        e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e
        for e in arrays
    ]
    weights = [
        e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e
        for e in weights
    ]

    if bins is None:
        bins = var_to_bins(title.strip(r"\\"))
        if bins is None:
            bins = binborders_wo_outliers(arrays[0])
    n_bins = len(bins) - 1

    hists = [
        np.histogram(arr, bins=bins, weights=w)[0]
        for arr, w in zip(arrays, weights)
    ]
    if weights[0] is None:
        errors = [np.sqrt(hist) for hist in hists]
    else:
        weights = [e / np.sum(e) for e in weights]
        errors = [
            np.sqrt(np.histogram(arr, bins=bins, weights=w**2)[0])
            for arr, w in zip(arrays, weights)
        ]

    scale_factor = 0  # int(np.floor(np.log10(max(sim_hist.max(), gen_hist.max()))))

    for i in range(len(hists)):
        hists[i] = hists[i] * (10**-scale_factor)
        errors[i] = errors[i] * (10**-scale_factor)

    sim_hist = hists[0]
    sim_error = errors[0]

    ax: Axes
    axrat: Axes
    fig, (ax, axrat) = plt.subplots(
        2,
        1,
        figsize=(5.5, 6),
        gridspec_kw={"height_ratios": [2, 0.7]},
        sharex="col",
    )

    artists = mplhep.histplot(
        np.stack(hists),
        bins=bins,
        label=labels,  # Use custom labels
        yerr=np.stack(errors) if weights[0] is None else None,
        ax=ax,
    )

    # overflow bins
    delta = (bins[1] - bins[0]) / 2
    colors = [a.stairs.get_edgecolor() for a in artists]
    print("foo")
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
                ymax=undershot * (10**-scale_factor),
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
                ymax=overshot * (10**-scale_factor),
                color=color,
                **kwstyle,
            )
            ibar += 1

    if (sim_hist > (sim_hist.max() / 10)).mean() < 0.1:
        ax.set_yscale("log")
    else:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_fontsize(13)

    axrat.set_ylim(0.48, 1.52)
    for gen_hist, gen_error, label, color in zip(
        hists[1:], errors[1:], labels[1:], colors[1:]
    ):
        if len(arrays) == 2:
            color = "black"
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = gen_hist / sim_hist
            frac_error_y = np.abs(frac) * np.sqrt(
                (sim_error / sim_hist) ** 2 + (gen_error / gen_hist) ** 2
            )
            frac_error_x = np.array([(bins[1] - bins[0]) / 2.0] * n_bins)
            frac_mask = (frac != 0) & np.invert(np.isnan(frac_error_y))

        x = bincenters(bins)

        axrat.errorbar(
            x=x[frac_mask],
            y=frac[frac_mask],
            yerr=frac_error_y[frac_mask],
            xerr=frac_error_x[frac_mask],
            barsabove=True,
            linestyle="",
            marker=None,
            ecolor=color,  # Use different color for each dataset
            label=label,  # Add label
            markersize=2,
        )
        # Indicators for out of range ratios:
        lower, upper = axrat.get_ylim()

        def ax_to_data(tup):
            tup = axrat.transAxes.transform(tup)
            tup = axrat.transData.inverted().transform(tup)
            return tup

        yupper = ax_to_data((0, 0.95))[1]
        ylower = ax_to_data((0, 0.05))[1]

        oor_idxs = np.where(frac > upper)[0]
        axrat.plot(
            x[oor_idxs],
            np.ones_like(oor_idxs) * yupper,
            "^",
            color=color,
            markersize=1.5,
        )

        oor_idxs = np.where(frac < lower)[0]
        axrat.plot(
            x[oor_idxs],
            np.ones_like(oor_idxs) * ylower,
            "v",
            color=color,
            markersize=1.5,
        )

    ax.set_ylabel("Counts/Bin", fontsize=17)
    ax.legend(fontsize=14, loc="best", framealpha=1, edgecolor="black")

    if len(arrays) > 2:
        axrat.set_ylabel(r"$\frac{\text{Dataset}}{\text{Simulation}}$", fontsize=15)
    else:
        axrat.set_ylabel(r"$\frac{\text{Model}}{\text{Simulation}}$", fontsize=15)

    # horizontal grid
    ax.grid(True, "major", "y")
    # grey r==1 line
    axrat.axhline(1, color="grey")

    # bounding box around the plots
    for iax in [ax, axrat]:
        for spline in iax.spines.values():
            spline.set_linewidth(1)
            spline.set_color("black")
    fig.suptitle(title, fontsize=26)

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
    # fig.savefig("/home/mscham/fgsim/wd/tmp.pdf")
    return fig
