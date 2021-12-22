import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog


def test_procedure() -> None:
    holder: Holder = Holder()
    train_log: TrainLog = holder.train_log

    loader: QueuedDataLoader = QueuedDataLoader()

    # if not experiment.ended:
    #     logger.error("Training has not completed, stopping.")
    #     loader.qfseq.stop()
    #     exit(0)

    # holder.select_best_model()
    # holder.models.eval()

    holder.models.eval()
    # Make sure the batches are loaded
    _ = loader.testing_batches
    loader.qfseq.stop()

    vals = {"gen": {}, "true": {}}

    # Iterate over the validation sample
    for batch in tqdm(loader.testing_batches, postfix="testing"):
        with torch.no_grad():
            batch = batch.clone().to(device)
            for key, nparr in batch.hlvs.items():
                if key not in vals["true"]:
                    vals["true"][key] = []
                vals["true"][key].append(list(nparr))

            holder.reset_gen_points()
            holder.gen_points.compute_hlvs()
            for key, nparr in holder.gen_points.hlvs.items():
                if key not in vals["gen"]:
                    vals["gen"][key] = []
                vals["gen"][key].append(list(nparr))

    for var in vals["true"]:
        vals["true"][var] = (
            torch.tensor(vals["true"][var]).flatten().detach().numpy()
        )
        vals["gen"][var] = torch.tensor(vals["gen"][var]).flatten().detach().numpy()

    for var in vals["gen"]:
        figure = plot2d(var, vals["true"][var], vals["gen"][var])
        with train_log.experiment.test():
            train_log.experiment.log_figure(
                figure_name=f"test-distplot-{var}", figure=figure, overwrite=True
            )
    logger.info("Done with batches.")
    exit(0)


plotconf = dict(hist_kws={"alpha": 0.6}, kde_kws={"linewidth": 2})


def plot2d(var, xtrue, xgen):
    logger.info(f"Plotting  var {var} {len(xtrue)} {len(xgen)}")
    fig = plt.figure(figsize=(10, 7))

    sns.histplot(
        xtrue,
        color="dodgerblue",
        label=f"simulated μ ({np.mean(xtrue)}) σ ({np.std(xtrue)}) ",
        alpha=0.6,
    )

    sns.histplot(
        xgen,
        color="orange",
        label=f"generated μ ({np.mean(xgen)}) σ ({np.std(xgen)}) ",
        alpha=0.6,
    )

    plt.title(var)
    plt.legend()
    from pathlib import Path

    path = Path(f"{conf.path.run_path}/diffplots/")
    path.mkdir(exist_ok=True)
    outputpath = path / f"{var}.pdf"

    plt.savefig(outputpath)

    return fig
