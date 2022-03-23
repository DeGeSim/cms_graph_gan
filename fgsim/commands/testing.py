"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import kstest
from torch_scatter import scatter_mean
from tqdm import tqdm

from fgsim.config import conf
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.io.sel_seq import batch_tools
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.plot.hlv_marginals import hlv_marginals
from fgsim.plot.xyscatter import xyscatter


def convert(tensorarr: List[torch.Tensor]) -> np.ndarray:
    return torch.cat(tensorarr).flatten().detach().cpu().numpy()


def subsample(arr: np.ndarray):
    return np.random.choice(arr, size=conf.testing.n_events, replace=False)


def get_testing_datasets(holder: Holder):
    # Make sure the batches are loaded
    loader: QueuedDataLoader = QueuedDataLoader()
    loader.qfseq.stop()
    # Sample at least 2k events
    n_batches = int(conf.testing.n_events / conf.loader.batch_size)
    assert n_batches <= len(loader.testing_batches)
    ds_dict = {"sim": loader.testing_batches[:n_batches]}

    for best_or_last in ["best", "last"]:
        ds_path = Path(conf.path.run_path) / f"test_{best_or_last}.pt"
        ds_is_available = ds_path.is_file()

        if best_or_last == "best":
            step = holder.state.best_grad_step
        else:
            step = holder.state.grad_step

        # Check if we need to rerun the model
        if ds_is_available:
            unp = torch.load(ds_path)
            if unp["step"] != step:
                ds_is_available = False
            elif len(unp["batches"]) != n_batches:
                ds_is_available = False
            else:
                gen_batches = unp["batches"]
        # if yes, pickle it
        if not ds_is_available:
            if best_or_last == "best":
                holder.select_best_model()

            holder.models.eval()

            gen_batches = []
            # generate a batch for each simulated one:
            for _ in tqdm(
                range(n_batches), desc=f"Generating Batches {best_or_last}"
            ):
                with torch.no_grad():
                    holder.reset_gen_points()
                    gen_batch = holder.gen_points.clone().cpu()
                    gen_batches.append(gen_batch)

            # with Pool(10) as p:
            #     gen_batches =
            # list(tqdm(p.imap(batch_tools.batch_compute_hlvs, gen_batches))
            gen_batches = [
                batch_tools.batch_compute_hlvs(batch)
                for batch in tqdm(gen_batches, desc="Compute HLVs gen_batches")
            ]

            torch.save({"step": step, "batches": gen_batches}, ds_path)
        ds_dict[best_or_last] = gen_batches

    return ds_dict["sim"], ds_dict["best"], ds_dict["last"]


def test_procedure() -> None:
    holder: Holder = Holder()
    train_log: TrainLog = holder.train_log

    # if not train_log.experiment.ended:
    #     logger.error("Training has not completed, stopping.")
    #     loader.qfseq.stop()
    #     exit(0)
    sim_batches, gen_batches_best, gen_batches_last = get_testing_datasets(holder)

    for best_or_last in ["best", "last"]:
        plot_path = Path(f"{conf.path.run_path}/plots_{best_or_last}/")
        plot_path.mkdir(exist_ok=True)
        gen_batches = (
            gen_batches_best if best_or_last == "best" else gen_batches_last
        )

        vals: Dict[str, Dict[str, List[torch.Tensor]]] = {"gen": {}, "sim": {}}

        for sim_batch, gen_batch in zip(sim_batches, gen_batches):
            for key in gen_batch.hlvs:
                if key not in vals["sim"]:
                    vals["sim"][key] = []
                if key not in vals["gen"]:
                    vals["gen"][key] = []
                vals["sim"][key].append(sim_batch.hlvs[key])
                vals["gen"][key].append(gen_batch.hlvs[key])

        # # Iterate over the test sample
        # for ibatch, batch in enumerate(tqdm(sim_batches, postfix="testing")):
        #     with torch.no_grad():
        #         batch = batch.clone().to(device)
        #         holder.reset_gen_points()
        #         holder.gen_points.compute_hlvs()
        #         for key in holder.gen_points.hlvs:
        #             if key not in vals["sim"]:
        #                 vals["sim"][key] = []
        #             if key not in vals["gen"]:
        #                 vals["gen"][key] = []
        #             vals["sim"][key].append(batch.hlvs[key])
        #             vals["gen"][key].append(holder.gen_points.hlvs[key])

        # Merge the computed hlvs from all batches

        sample: Dict[str, Dict[str, np.ndarray]] = {
            sim_or_gen: {
                var: subsample(convert(vals[sim_or_gen][var]))
                for var in vals[sim_or_gen]
            }
            for sim_or_gen in ("sim", "gen")
        }
        # KS tests
        for var in sample["gen"].keys():
            res = kstest(sample["sim"][var], sample["gen"][var])
            with train_log.experiment.test():
                train_log.experiment.log_metric(f"kstest-{var}", res.pvalue)

        # Scatterplots
        logger.info(f"Plotting  xyscatter")
        # Scatter of a single event
        figure = xyscatter(
            sim=sim_batches[0][0].x.numpy(),
            gen=gen_batches[0][0].x.numpy(),
            outputpath=plot_path / f"xyscatter_single.pdf",
            title="Scatter event",
        )
        with train_log.experiment.test():
            train_log.experiment.log_figure(
                figure_name=f"test-xyscatter", figure=figure, overwrite=True
            )

        figure = xyscatter(
            sim=sim_batches[0].x.numpy(),
            gen=gen_batches[0].x.numpy(),
            outputpath=plot_path / f"xyscatter_batch.pdf",
            title="Scatter all points in batch",
        )

        figure = xyscatter(
            sim=scatter_mean(sim_batches[0].x, sim_batches[0].batch, dim=0).numpy(),
            gen=scatter_mean(gen_batches[0].x, gen_batches[0].batch, dim=0).numpy(),
            outputpath=plot_path / f"xyscatter_batch_means.pdf",
            title=f"Event means for a batch ({len(sim_batches[0].batch)})",
        )
        with train_log.experiment.test():
            train_log.experiment.log_figure(
                figure_name=f"xyscatter_batch_means", figure=figure, overwrite=True
            )

        # # Sample 2k events and plot the distribution
        for var in sample["gen"]:
            xsim = sample["sim"][var]
            xgen = sample["gen"][var]
            logger.info(f"Plotting  var {var}")
            figure = hlv_marginals(
                var, xsim=xsim, xgen=xgen, outputpath=plot_path / f"{var}.pdf"
            )
            with train_log.experiment.test():
                train_log.experiment.log_figure(
                    figure_name=f"distplot-{var}", figure=figure, overwrite=True
                )

    exit(0)
