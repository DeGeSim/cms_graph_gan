# %%
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import binned_statistic

os.chdir("/home/mscham/fgsim")

# %%
# Calculate the errors


testds = torch.load("data/clic_valtest/validation.torch")
# %%
evals = torch.cat([batch.energy for batch in testds]).numpy()
errors = []
for lower in np.arange(0, 500, 10):
    upper = lower + 10
    in_bin = evals[(lower < evals) & (evals < upper)]
    error = np.std(in_bin) / np.sqrt(len(in_bin)) / np.mean(in_bin) * 100
    errors.append(error)

# %%

filepaths = glob("wd/*/*/prediction.csv")

confL = [
    OmegaConf.load(fp.rstrip("prediction.csv") + "resulting_train_config.yaml")
    for fp in filepaths
]
dfL = [pd.read_csv(fp) for fp in filepaths]

tag_to_label_dict = {
    "deepconv/dnn/msgpass": "GNN (7.5k)",
    "deepconv/dnn/hlv/msgpass": "GNN+HLV (19k)",
    "linreg": "LinReg (3)",
    "hlv/linreg": "DNN(HLV) (3.3k)",
    "cnn": "GoogLeNet* (15M)",
}

tag_to_color = {
    "deepconv/dnn/msgpass": 1,
    "deepconv/dnn/hlv/msgpass": 2,
    "linreg": 3,
    "hlv/linreg": 4,
    "cnn": 5,
}
mplcolors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
# %%
# Error plot
bins = np.arange(0, 510, 10)
bincenters = np.arange(5, 505, 10)
# bincenters = (bins[1:] + bins[:-1]) / 2

plt.figure(num=None, figsize=(9, 5), dpi=80, facecolor="w", edgecolor="k")

tags = [
    "linreg",
    "hlv/linreg",
    "deepconv/dnn/msgpass",
    "deepconv/dnn/hlv/msgpass",
    "cnn",
]

for tag in tags:
    for c, d in zip(confL, dfL):
        tmptags = c.tag.split("_")
        tmptags.sort()
        newtag = "/".join(tmptags)
        if tag == newtag:
            conf = c
            df = d
            print(f"found tag {tag}")

    stat, edges, binnumber = binned_statistic(
        df["Energy"], df["Relativ Error"] * 100, "mean", bins=bins
    )

    xL = [bincenters[ibin] for ibin in range(len(bincenters))]
    yL = np.array([stat[ibin] for ibin in range(len(bincenters))])
    # plt.errorbar(xL, yL, yerr=np.array(errors), label=tag_to_label_dict[tag])
    plt.plot(
        xL,
        yL,
        color=mplcolors[tag_to_color[tag]],
        label=tag_to_label_dict[tag],
        lw=2,
    )
    plt.fill_between(
        xL,
        yL - errors,
        yL + errors,
        color=mplcolors[tag_to_color[tag]],
        interpolate=True,
        alpha=0.4,
    )
plt.yscale("log")
plt.ylim(5e-1, 5e1)
plt.xlim(0, 500)
plt.grid(True, "both", "both")
plt.xlabel("Energy [GeV]")
plt.ylabel("Relativ Error [%]")
plt.title(f"Relativ Error vs True Energy on the test set")
plt.legend()
plt.savefig(f"wd/rel_error_vs_true_energy.pdf")

plt.show()
# %%
# pdgset = set()
# theta = set()
# phi = set()
# for fp in glob("data/clic/*.h5"):
#     with h5py.File(fp) as f:
#         pdgset |= set(np.unique(f["pdgID"][:]))
#         theta |= set(np.unique(f["theta"][:]))
#         phi |= set(np.unique(f["phi"][:]))
# logger.info("pdgset", pdgset)
# logger.info("theta", theta)
# logger.info("phi", phi)
# %%
# To big
# plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor="w", edgecolor="k")
# for conf, df in zip(confL, dfL):
#     plt.scatter(
#         df["Energy"],
#         df["Prediction"],
#         label=conf.model.name,
#         alpha=0.1,
#         s=12,
#     )
# plt.plot([10, 510], [10, 510], label="Ideal", color="red")
# plt.xlabel("Enery")
# plt.ylabel("Prediction")
# plt.title(f"Prediction vs True Energy")
# plt.legend()
# plt.savefig(f"wd/prediction_vs_true_energy.pdf")


# %%
