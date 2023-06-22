import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def get_grad_dict(model):
    named_parameters = model.named_parameters()
    ave_grads = []
    # max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(
                n.rstrip("_orig")
                .rstrip(".weight")
                .rstrip(".linear")
                .replace(".seq.", ".")
            )
            ave_grads.append(p.grad.abs().mean().cpu())
            # max_grads.append(p.grad.abs().max().cpu())
    return {k: v for k, v in zip(layers, ave_grads)}


def fig_grads(graddict, partname) -> plt.Figure:
    # plt.title(f"{var}, all gen values are nan")
    layers = list(graddict.keys())
    ave_grads = list(graddict.values())

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
    nparts = len(ave_grads)
    timesteps = len(ave_grads[0])
    xaxis = np.arange(nparts)

    fig = plt.figure(figsize=(20 + timesteps, 24))
    plt.cla()
    plt.clf()
    plt.imshow(
        ave_grads,
        cmap=plt.cm.coolwarm,
        norm=LogNorm(),
    )
    plt.yticks(ticks=xaxis, labels=layers_formated, family="monospace")
    plt.xticks(np.arange(timesteps))
    plt.ylabel("Layers")
    plt.xlabel("Time")
    plt.colorbar()
    plt.title(f"{partname} Gradient Scale")
    plt.tight_layout()
    # plt.savefig("wd/modelgrads.pdf")

    return fig
