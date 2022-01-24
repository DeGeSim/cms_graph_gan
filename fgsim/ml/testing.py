"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from typing import Dict, List

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


def convert(tensorarr: List[torch.Tensor]) -> np.ndarray:
    return torch.cat(tensorarr).flatten().detach().cpu().numpy()


def subsample(arr: np.ndarray):
    return np.random.choice(arr, size=conf.testing.n_events, replace=False)


def test_procedure() -> None:
    holder: Holder = Holder()
    # holder.select_best_model()
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

    vals: Dict[str, Dict[str, List[torch.Tensor]]] = {"gen": {}, "sim": {}}

    # Iterate over the test sample
    for ibatch, batch in enumerate(tqdm(loader.testing_batches, postfix="testing")):
        with torch.no_grad():
            batch = batch.clone().to(device)
            holder.reset_gen_points()
            holder.gen_points.compute_hlvs()
            for key in holder.gen_points.hlvs:
                if key not in vals["sim"]:
                    vals["sim"][key] = []
                if key not in vals["gen"]:
                    vals["gen"][key] = []
                vals["sim"][key].append(batch.hlvs[key])
                vals["gen"][key].append(holder.gen_points.hlvs[key])
        # Sample at least 2k events
        if ibatch >= (conf.testing.n_events / conf.loader.batch_size):
            break

    # Merge the computed hlvs from all batches

    sample: Dict[str, Dict[str, np.ndarray]] = {
        sim_or_gen: {
            var: subsample(convert(vals[sim_or_gen][var]))
            for var in vals[sim_or_gen]
        }
        for sim_or_gen in ("sim", "gen")
    }

    for var in sample["gen"].keys():
        res = kstest(sample["sim"][var], sample["gen"][var])
        with train_log.experiment.test():
            train_log.experiment.log_metric(f"kstest-{var}", res.pvalue)

    # # Sample 2k events and plot the distribution
    for var in sample["gen"]:
        xsim = sample["sim"][var]
        xgen = sample["gen"][var]
        logger.info(f"Plotting  var {var}")
        figure = diffhist(var, xsim=xsim, xgen=xgen)
        with train_log.experiment.test():
            train_log.experiment.log_figure(
                figure_name=f"test-distplot-{var}", figure=figure, overwrite=True
            )

    exit(0)
