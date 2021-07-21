# %%
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import binned_statistic

os.chdir("/home/mscham/fgsim")

filepaths = glob("wd/*/*/prediction.csv")

confL = [
    OmegaConf.load(fp.rstrip("prediction.csv") + "resulting_train_config.yaml")
    for fp in filepaths
]
dfL = [pd.read_csv(fp) for fp in filepaths]
# %%
plt.figure(num=None, figsize=(10, 5), dpi=80, facecolor="w", edgecolor="k")
for conf, df in zip(confL, dfL):
    plt.scatter(
        df["Energy"],
        df["Prediction"],
        label=conf.model.name,
        alpha=0.1,
        s=12,
    )
plt.plot([10, 510], [10, 510], label="Ideal", color="red")
plt.xlabel("Enery")
plt.ylabel("Prediction")
plt.title(f"Prediction vs True Energy")
plt.legend()
plt.savefig(f"wd/prediction_vs_true_energy.pdf")

# %%
# Error plot


bins = np.array(range(10, 520, 10))
bincenters = (bins[1:] + bins[:-1]) / 2

plt.figure(num=None, figsize=(6, 5), dpi=80, facecolor="w", edgecolor="k")
for conf, df in zip(confL, dfL):
    stat, edges, binnumber = binned_statistic(
        df["Energy"], df["Relativ Error"] * 100, "mean", bins=bins
    )

    # plt.plot(
    #     bincenters,
    #     stat,
    #     label=conf.model.name,
    #     alpha=0.5,
    # )
    binpos, count = np.unique(binnumber - 1, return_counts=True)

    xL = []
    yL = []
    errorL = []
    skippedbins = 0
    for ibin in range(len(bincenters)):
        if binpos[ibin - skippedbins] != ibin:
            skippedbins += 1
            continue
        xL.append(bincenters[ibin])
        yL.append(stat[ibin])
        errorL.append(1 / np.sqrt(count[ibin - skippedbins]))
    plt.errorbar(xL, yL, yerr=np.array(errorL), label=conf.model.name)
plt.yscale("log")
plt.ylim(5e-1, 5e1)
plt.xlim(0, 500)
plt.grid(True, "both", "both")
plt.xlabel("Energy")
plt.ylabel("Relativ Error [%]")
plt.title(f"Relativ Error vs True Energy")
plt.legend()
plt.savefig(f"wd/rel_error_vs_true_energy.pdf")
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
