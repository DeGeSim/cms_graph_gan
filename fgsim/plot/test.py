from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fgsim.config import conf


def diffhist(var, xsim: np.array, xgen: np.array) -> plt.Figure:
    assert xsim.shape == xgen.shape
    fullarr = np.array([xsim, xgen]).flatten()
    fullarr.sort()

    be = np.histogram_bin_edges(fullarr, bins="fd")
    hist = np.histogram(fullarr, bins=be)[0]

    minevents = min([len(e) / 500 for e in (xsim, xgen)])

    # Merge bins with less then two events on the end of the spectrum
    be_list = list(be)

    hstart = 0
    while hist[hstart] < minevents:
        be_list.pop(1)
        hstart += 1

    removed_end = 0
    hend = len(hist) - 1
    while hist[hend] < minevents:
        be_list.pop(-2)
        hend -= 1
        removed_end += 1

    # start and end must be the same
    assert be_list[0] == be[0]
    assert be_list[-1] == be[-1]
    # hstart counts the number of removed elements
    assert be_list[1] == be[hstart + 1]
    # number of removed elements
    assert be_list[-2] == be[-2 - removed_end]

    fig = plt.figure(figsize=(10, 7))
    sns.histplot(
        {
            f"simulated μ ({np.mean(xsim):.2E}) σ ({np.std(xsim):.2E}) ": xsim,
            f"generated μ ({np.mean(xgen):.2E}) σ ({np.std(xgen):.2E}) ": xgen,
        },
        bins=be[hstart : hend + 1],
        alpha=0.6,
        legend=True,
    )
    plt.title(var)

    path = Path(f"{conf.path.run_path}/diffplots/")
    path.mkdir(exist_ok=True)
    outputpath = path / f"{var}.pdf"

    plt.savefig(outputpath)

    return fig
