import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure


def get_grad_dict(model):
    named_parameters = model.named_parameters()
    ave_grads = []
    # max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(
                n.replace(".parametrizations.weight.orig", ".snorm")
                .rstrip("_orig")
                .rstrip("_weight")
                .replace(".seq.", ".")
                .replace(".nn.", ".")
                .replace("_nn.", ".")
                .replace(".reduction.", ".red.")
                .rstrip(".linear")
            )
            ave_grads.append(p.grad.abs().mean().cpu())
            # max_grads.append(p.grad.abs().max().cpu())
    return {k: v for k, v in zip(layers, ave_grads)}


def fig_grads(grad_aggr, partname: str) -> Figure:
    graddict: OrderedDict = grad_aggr.history

    # plt.title(f"{var}, all gen values are nan")
    steps = np.array(grad_aggr.steps)
    layers = list(graddict.keys())
    ave_grads = np.array([list(e) for e in graddict.values()])

    layers_split = [e.split(".") for e in layers]
    max_levels = max([len(e) for e in layers_split])
    # expand to  same size
    layers_split = [e + [""] * (max_levels - len(e)) for e in layers_split]
    max_chars_per_lvl = [
        max([len(e[ilevel]) for e in layers_split]) for ilevel in range(max_levels)
    ]
    # padd the characters
    layers_split = [
        [
            substr + " " * (max_chars_per_lvl[ilvl] - len(substr))
            for ilvl, substr in enumerate(e)
        ]
        for e in layers_split
    ]
    layers_formated = ["/".join(e) for e in layers_split]

    max_bins = 50

    ave_grads = pad_to_multiple(ave_grads, max_bins)
    steps = pad_to_multiple(steps.reshape(1, -1), max_bins).reshape(-1)
    nparts, ntimesteps = ave_grads.shape

    plt.clf()
    fig = plt.figure(figsize=(24, 24))

    plt.imshow(ave_grads, cmap=plt.cm.coolwarm, norm=LogNorm())
    plt.yticks(ticks=np.arange(nparts), labels=layers_formated, family="monospace")
    plt.xticks(ticks=np.arange(ntimesteps), labels=steps, rotation=45)
    plt.ylabel("Layers")
    plt.xlabel("Step")
    plt.colorbar()
    plt.title(f"{partname} Gradient Scale")
    plt.tight_layout()
    # plt.savefig("wd/modelgrads.pdf")

    return fig


def pad_to_multiple(arr: np.ndarray, max_bins: int):
    nparts, nentries = arr.shape
    if nentries <= max_bins:
        return arr
    pad_to = math.ceil(nentries / max_bins) * max_bins
    arr = np.pad(
        arr,
        pad_width=((0, 0), (0, pad_to - nentries)),
        constant_values=np.nan,
    )
    return arr.reshape(nparts, -1, max_bins)[..., 0]
