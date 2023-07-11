from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure

from fgsim.plot.binborders import binborders_wo_outliers, chip_to_binborders
from fgsim.utils.torchtonp import wrap_torch_to_np

np.set_printoptions(formatter={"float_kind": "{:.3g}".format})


@wrap_torch_to_np
def hist2d(
    sim: np.ndarray,
    gen: np.ndarray,
    title: str,
    v1name: str,
    v2name: str,
    v1bins: Optional[np.ndarray] = None,
    v2bins: Optional[np.ndarray] = None,
    step: Optional[int] = None,
) -> Figure:
    plt.cla()
    plt.clf()

    sns.set()
    fig: Figure
    sim_axes: Axes
    gen_axes: Axes
    fig, (sim_axes, gen_axes) = plt.subplots(
        1, 2, sharex=True, sharey=True  # , figsize=(20, 12)
    )
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

    shist, _, _ = np.histogram2d(
        chip_to_binborders(sim[:, 0], xedges),
        chip_to_binborders(sim[:, 1], yedges),
        bins=(xedges, yedges),
    )  # , density=density, weights=weights)
    ghist, _, _ = np.histogram2d(
        chip_to_binborders(gen[:, 0], xedges),
        chip_to_binborders(gen[:, 1], yedges),
        bins=(xedges, yedges),
    )

    if (shist > (shist.max() / 10)).mean() < 0.1:
        norm = LogNorm(max(shist.min(), 1), shist.max())
    else:
        norm = Normalize(shist.min(), shist.max())

    for iax, (ax, hist) in enumerate(zip([sim_axes, gen_axes], [shist, ghist])):
        im = ax.imshow(hist.T, cmap=plt.cm.hot, norm=norm, aspect="auto")
        if iax == 0:
            ax.set_ylabel(v2name)
            ax.set_yticks(**edge_to_labels(yedges))
        ax.set_xticks(rotation=45, **edge_to_labels(xedges))
        ax.set_xlabel(v1name)

    # s_cax = sim_axes.inset_axes([1.04, 0.2, 0.05, 0.6])
    fig.colorbar(im)

    sim_axes.set_title("Simulation")
    gen_axes.set_title("Model")

    if step is not None:
        title += f"\nStep {step}"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def edge_to_labels(edges):
    if len(edges) < 10:
        ticks = np.arange(len(edges))
    else:
        ticks = np.linspace(0, len(edges) - 1, 10, dtype="int")
    labels = [str(int(e)) for e in edges[ticks]]
    return dict(ticks=ticks, labels=labels)
