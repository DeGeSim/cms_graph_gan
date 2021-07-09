# %%
# %%
# %cd ~/fgsim
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ColorConverter
from matplotlib.pyplot import figure
from scipy.interpolate import make_interp_spline
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# %%
event_acc = EventAccumulator("runs/May20_11-50-09_max-wgse002.desy.de")
event_acc.Reload()
# Show all tags in the log file
tags = event_acc.Tags()["scalars"]

dD = {tag: [e.value for e in event_acc.Scalars(tag)] for tag in tags}
df = pd.DataFrame(dD)
# %%
for tag in ["loss", "nndiff", "simplediff"]:
    y = df[tag]
    figure(num=None, figsize=(10, 5), dpi=80, facecolor="w", edgecolor="k")
    x = np.arange(len(y))
    xnew = np.linspace(x.min(), x.max(), 100)
    spl = make_interp_spline(x, y, k=7)
    y_smooth = spl(xnew)
    plt.plot(y, color=ColorConverter().to_rgba("C0", 0.4))
    plt.plot(xnew, y_smooth, color="C1", label="smoothed")
    plt.title(
        {
            "loss": "Mean Squared Error",
            "nndiff": "Error of the NN in %",
            "simplediff": "Error for Sum over the cell energies",
        }[tag]
    )
    if tag == "loss":
        plt.yscale("log")
    plt.legend()
    plt.savefig(f"wd/forward/{tag}.pdf")
    plt.show()


# %%
