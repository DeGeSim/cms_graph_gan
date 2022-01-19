"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

import numpy as np
import torch
from scipy.stats import kstest
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.plot.test import diffhist


def test_procedure() -> None:
    holder: Holder = Holder()
    holder.select_best_model()
    holder.models.eval()

    train_log: TrainLog = holder.train_log

    loader: QueuedDataLoader = QueuedDataLoader()

    # if not experiment.ended:
    #     logger.error("Training has not completed, stopping.")
    #     loader.qfseq.stop()
    #     exit(0)

    # Make sure the batches are loaded
    _ = loader.testing_batches
    loader.qfseq.stop()

    vals = {"gen": {}, "sim": {}}

    # Iterate over the test sample
    for ibatch, batch in enumerate(tqdm(loader.testing_batches, postfix="testing")):
        with torch.no_grad():
            batch = batch.clone().to(device)
            for key, nparr in batch.hlvs.items():
                if key not in vals["sim"]:
                    vals["sim"][key] = []
                vals["sim"][key].append(list(nparr))

            holder.reset_gen_points()
            holder.gen_points.compute_hlvs()
            for key, nparr in holder.gen_points.hlvs.items():
                if key not in vals["gen"]:
                    vals["gen"][key] = []
                vals["gen"][key].append(list(nparr))
        # Sample at least 2k events
        if ibatch >= (2000 / conf.loader.batch_size):
            break

    # Merge the computed hlvs from all batches
    for var in vals["sim"]:
        vals["sim"][var] = torch.tensor(vals["sim"][var]).flatten().detach().numpy()
        vals["gen"][var] = torch.tensor(vals["gen"][var]).flatten().detach().numpy()

    for var in vals["gen"].keys():
        res = kstest(subsample(vals["sim"][var]), subsample(vals["gen"][var]))
        with train_log.experiment.test():
            train_log.experiment.log_metric(f"kstest-{var}", res.pvalue)

    # # Sample 2k events and plot the distribution
    for var in vals["gen"]:
        xsim = subsample(vals["sim"][var])
        xgen = subsample(vals["gen"][var])
        logger.info(f"Plotting  var {var}")
        figure = diffhist(var, xsim=xsim, xgen=xgen)
        with train_log.experiment.test():
            train_log.experiment.log_figure(
                figure_name=f"test-distplot-{var}", figure=figure, overwrite=True
            )

    exit(0)


def subsample(arr: np.ndarray):
    return np.random.choice(arr, size=2000, replace=False)
