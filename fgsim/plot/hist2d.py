from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa
from matplotlib import colors, ticker
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from fgsim.plot.binborders import binborders_wo_outliers, chip_to_binborders

np.set_printoptions(formatter={"float_kind": "{:.3g}".format})


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


def hist2d(
    sim: np.ndarray,
    gen: np.ndarray,
    v1name: str,
    v2name: str,
    v1bins: Optional[np.ndarray] = None,
    v2bins: Optional[np.ndarray] = None,
    simw: Optional[np.ndarray] = None,
    genw: Optional[np.ndarray] = None,
    colorlim: Optional[float] = None,
    colorlim_delta: Optional[float] = None,
    force_log=False,
    force_log_delta=False,
    cbtitle=None,
    # step: Optional[int] = None,
) -> Figure:
    plt.cla()
    plt.clf()

    if len(sim) > 500_000:
        sel = np.random.choice(sim.shape[0], 500_000)
        sim = sim[sel]
        if simw is not None:
            simw = simw * (simw.shape[0] / 500_000)
            simw = simw[sel]
    if len(gen) > 500_000:
        sel = np.random.choice(gen.shape[0], 500_000)
        gen = gen[sel]
        if genw is not None:
            genw = genw * (genw.shape[0] / 500_000)
            genw = genw[sel]

    if v1bins is None:
        xedges = binborders_wo_outliers(sim[:, 0])
        yedges = binborders_wo_outliers(sim[:, 1])
    else:
        xedges = v1bins
        yedges = v2bins
    # flip x and y axis if the x val has more bins
    if len(xedges) > len(yedges) + 5:
        xedges, yedges = yedges, xedges
        sim = sim[:, (1, 0)]
        gen = gen[:, (1, 0)]
        v1bins, v2bins = v2bins, v1bins
        v1name, v2name = v2name, v1name

    pagewh = [448.13 / 72.0, 636.6 / 72.0]
    width, height = pagewh[0] * 1.3 / 3.0, pagewh[1] * 1.3 / 3.0
    fig, axs = plt.subplots(
        2, 3, figsize=(width * 3, height), height_ratios=[1, 0.05]
    )

    ax1: Axes = axs[0, 0]
    ax2: Axes = axs[0, 1]
    ax3: Axes = axs[0, 2]
    axs[1, 0].remove()
    axs[1, 1].remove()
    gs = axs[0, 0].get_gridspec()
    axcol: Axes = fig.add_subplot(gs[1, :2])
    axrcol: Axes = axs[1, 2]

    hists = []
    ax: Axes
    norm = "log" if force_log else None
    for iax, (ax, arr, weights, title) in enumerate(
        zip([ax1, ax2], [sim, gen], [simw, genw], (simlabel(), "\\dt"))
    ):
        x, y = arr.T
        x = chip_to_binborders(x, xedges)
        y = chip_to_binborders(y, yedges)
        h, mesh, norm = _2dhist_with_autonorm(
            ax=ax,
            x=x,
            y=y,
            bins=[xedges, yedges],
            norm=norm,
            linewidths=2,
            weights=weights,
            vmax=colorlim,
        )
        hists.append(h)
        # sns.kdeplot(
        #     x=x,
        #     y=y,
        #     ax=ax,
        #     levels=[0.1, 0.3, 0.6, 0.9],
        #     color="black",
        #     thresh=0.2,
        #     bw_adjust=1.0,
        # )
        ax.set_title(title)
        ax.set_xlabel(v1name)
        if iax == 0:
            ax.set_ylabel(v2name)
        else:
            ax.yaxis.set_ticklabels([])

    cbar1 = plt.colorbar(mesh, cax=axcol, orientation="horizontal")

    # Ratio hist
    # subtract the simulated sample from the generated sample
    h = hists[1] - hists[0]
    # overwrite bins with less then 100 enties in the simulation
    insig = (hists[0] * len(sim)) < 100
    owidx = np.where(insig)
    h[owidx[0], owidx[1]] = 0
    h[np.where(h == 0)] = np.NAN

    # limit of the delta colorbar = half max of simlated hist
    diff_lim = np.nanmax(np.abs(hists[0]))
    # diff_lim = max(np.nanmax(np.abs(h)), diff_lim)
    # switch to log if less than 5% of bins have at most 10% of the maximum enties
    # if force_log_delta or (np.abs(h) > (np.max(np.abs(h)) / 10)).mean() < 0.05:
    # if h.max() / max(np.median(h), 1) > 6:
    if False:
        norm = colors.SymLogNorm(
            linthresh=1, linscale=1.0, vmin=-diff_lim, vmax=diff_lim
        )
        log_locator = ticker.SymmetricalLogLocator(transform=norm._trf)
        log_locator.set_params(numticks=5)
    else:
        norm = colors.CenteredNorm(
            vcenter=0,
            halfrange=diff_lim if colorlim_delta is None else colorlim_delta,
        )

    mesh = ax3.imshow(
        h.T,
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        norm=norm,
        cmap="bwr",
        origin="lower",
    )

    cbar2 = plt.colorbar(
        mesh,
        cax=axrcol,
        orientation="horizontal",
        # ticks=log_locator if force_log_delta else None,
    )

    ax3.yaxis.set_ticklabels([])
    ax3.set_xlabel(v1name)
    ax3.set_title(f"\\dt - {simlabel()}")

    if cbtitle is not None:
        cbar1.set_label(cbtitle)
        cbar2.set_label("$Δ$ " + cbtitle)

    fig.tight_layout()
    # fig.savefig("/home/mscham/fgsim/wd/test.pdf")
    # print("done")
    return fig


def edge_to_labels(edges):
    if len(edges) < 10:
        ticks = np.arange(len(edges))
    else:
        ticks = np.linspace(0, len(edges) - 1, 10, dtype="int")
    labels = [str(int(e)) for e in edges[ticks]]
    return dict(ticks=ticks, labels=labels)


def _2dhist_with_autonorm(
    ax,
    x,
    y,
    bins,
    norm=None,
    range=None,
    density=False,
    weights=None,
    vmax=None,
    **kwargs,
):
    xedges, yedges = bins
    h, _, _ = np.histogram2d(
        x, y, bins=bins, range=range, density=density, weights=weights
    )
    if norm == "log":
        norm = LogNorm(1, h.max())
    elif norm is None:
        if (h > (h.max() / 10)).mean() < 0.05:
            # if h.max() / max(np.median(h), 1) > 6:
            norm = LogNorm(1, h.max())
        else:
            norm = Normalize(0, h.max())

    pc = ax.pcolormesh(xedges, yedges, h.T, norm=norm, **kwargs)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])

    return h, pc, norm
