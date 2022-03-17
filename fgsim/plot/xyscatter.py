from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def xyscatter(
    sim: np.array, gen: np.array, outputpath: Path, title: str
) -> plt.Figure:
    np.set_printoptions(formatter={"float_kind": "{:.3g}".format})
    mean_sim = np.around(np.mean(sim, axis=0), 2)
    cov_sim = str(np.around(np.cov(sim, rowvar=0), 2)).replace("\n", "")
    mean_gen = np.around(np.mean(gen, axis=0), 2)
    cov_gen = str(np.around(np.cov(gen, rowvar=0), 2)).replace("\n", "")
    sim_df = pd.DataFrame(
        {
            "x": sim[:, 0],
            "y": sim[:, 1],
            "cls": f"sim μ{mean_sim}\nσ{cov_sim}",
        }
    )
    gen_df = pd.DataFrame(
        {
            "x": gen[:, 0],
            "y": gen[:, 1],
            "cls": f"gen μ{mean_gen}\nσ{cov_gen}",
        }
    )
    df = pd.concat([sim_df, gen_df], ignore_index=True)

    plt.cla()
    plt.clf()
    g: sns.JointGrid = sns.jointplot(data=df, x="x", y="y", hue="cls", legend=False)
    g.fig.suptitle(title)
    g.figure
    # g.ax_joint.collections[0].set_alpha(0)
    # g.fig.tight_layout()
    g.figure.subplots_adjust(top=0.95)
    plt.legend(
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        labels=[f"gen μ{mean_gen}\nσ{cov_gen}", f"sim μ{mean_sim}\nσ{cov_sim}"],
    )
    plt.tight_layout()

    g.savefig(outputpath)
    g.savefig(outputpath.with_suffix(".png"))

    return g.figure
