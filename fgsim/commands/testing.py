"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import kstest, wasserstein_distance
from torch_scatter import scatter_mean
from tqdm import tqdm

from fgsim.config import conf
from fgsim.io.batch_tools import batch_from_pcs_list
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.io.sel_seq import batch_tools
from fgsim.ml.holder import Holder
from fgsim.models.branching.graph_tree import GraphTreeWrapper
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.plot.hlv_marginals import hlv_marginals
from fgsim.plot.xyscatter import xy_hist, xyscatter, xyscatter_faint


@dataclass
class TestInfo:
    train_log: TrainLog
    sim_batches: List
    gen_batches: List
    hlvs_dict: Dict[str, Dict[str, np.ndarray]]
    plot_path: Path
    step: int
    best_or_last: str


def test_procedure() -> None:
    holder: Holder = Holder()
    train_log: TrainLog = holder.train_log

    sim_batches, gen_batches_best, gen_batches_last = get_testing_datasets(holder)

    for best_or_last in ["best", "last"]:
        plot_path = Path(f"{conf.path.run_path}/plots_{best_or_last}/")
        plot_path.mkdir(exist_ok=True)

        if best_or_last == "best":
            step = holder.state.best_grad_step
            gen_batches = gen_batches_best
        else:
            step = holder.state.grad_step
            gen_batches = gen_batches_last

        hlvs_dict = compute_hlvs_dict(sim_batches, gen_batches)
        test_info = TestInfo(
            train_log=train_log,
            sim_batches=sim_batches,
            gen_batches=gen_batches,
            hlvs_dict=hlvs_dict,
            plot_path=plot_path,
            step=step,
            best_or_last=best_or_last,
        )

        test_plots(test_info)
        if best_or_last == "best":
            test_metrics(test_info)

    exit(0)


def get_testing_datasets(holder: Holder):
    # Make sure the batches are loaded
    loader: QueuedDataLoader = QueuedDataLoader()
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
                    holder.gen_points = holder.gen_points.to_pcs_batch()
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


def compute_hlvs_dict(sim_batches, gen_batches) -> Dict[str, Dict[str, np.ndarray]]:
    hlvs_dict_torch: Dict[str, Dict[str, List[torch.Tensor]]] = {
        "gen": {},
        "sim": {},
    }

    for sim_batch, gen_batch in zip(sim_batches, gen_batches):
        for key in gen_batch.hlvs:
            if key not in hlvs_dict_torch["sim"]:
                hlvs_dict_torch["sim"][key] = []
            if key not in hlvs_dict_torch["gen"]:
                hlvs_dict_torch["gen"][key] = []
            hlvs_dict_torch["sim"][key].append(sim_batch.hlvs[key])
            hlvs_dict_torch["gen"][key].append(gen_batch.hlvs[key])

    # Sample 2k events and plot the distribution
    # convert to numpy
    hlvs_dict: Dict[str, Dict[str, np.ndarray]] = {
        sim_or_gen: {
            var: subsample(convert(hlvs_dict_torch[sim_or_gen][var]))
            for var in hlvs_dict_torch[sim_or_gen]
        }
        for sim_or_gen in ("sim", "gen")
    }
    return hlvs_dict


def convert(tensorarr: List[torch.Tensor]) -> np.ndarray:
    return torch.cat(tensorarr).flatten().detach().cpu().numpy()


def subsample(arr: np.ndarray):
    return np.random.choice(arr, size=conf.testing.n_events, replace=False)


def test_metrics(test_info: TestInfo):
    train_log = test_info.train_log
    sim_batches = test_info.sim_batches
    gen_batches = test_info.gen_batches
    hlvs_dict = test_info.hlvs_dict
    step = test_info.step
    sim = sim_batches[0].x.numpy()
    gen = gen_batches[0].x.numpy()

    sim_means = scatter_mean(sim_batches[0].x, sim_batches[0].batch, dim=0).numpy()
    gen_means = scatter_mean(gen_batches[0].x, gen_batches[0].batch, dim=0).numpy()

    # compute a covar matrix for each batch
    # take the sqrt of the elements to scale to the scale of the variable
    # and then compare the distributions with w1

    covars_sim = torch.vstack(
        [torch.cov(batch.x.T).reshape(1, 4) for batch in sim_batches]
    ).numpy()
    covars_gen = torch.vstack(
        [torch.cov(batch.x.T).reshape(1, 4) for batch in gen_batches]
    ).numpy()

    metrics_dict = {
        "w1_x": wasserstein_distance(sim[:, 0], gen[:, 0]),
        "w1_y": wasserstein_distance(sim[:, 1], gen[:, 1]),
        "w1_x_means": wasserstein_distance(sim_means[:, 0], gen_means[:, 0]),
        "w1_y_means": wasserstein_distance(sim_means[:, 1], gen_means[:, 1]),
        "w1_cov_xx": wasserstein_distance(covars_sim[:, 0], covars_gen[:, 0]),
        "w1_cov_xy": wasserstein_distance(covars_sim[:, 1], covars_gen[:, 1]),
        "w1_cov_yx": wasserstein_distance(covars_sim[:, 2], covars_gen[:, 2]),
        "w1_cov_yy": wasserstein_distance(covars_sim[:, 3], covars_gen[:, 3]),
    }
    # KS tests
    for var in hlvs_dict["gen"].keys():
        res = kstest(hlvs_dict["sim"][var], hlvs_dict["gen"][var])
        with train_log.experiment.test():
            train_log.experiment.log_metric(f"kstest-{var}", res.pvalue)
    with train_log.experiment.test():
        train_log.experiment.log_metrics(
            metrics_dict,
            step=step,
        )


def test_plots(test_info: TestInfo):
    train_log = test_info.train_log
    sim_batches = test_info.sim_batches
    gen_batches = test_info.gen_batches
    hlvs_dict = test_info.hlvs_dict
    plot_path = test_info.plot_path
    best_or_last = test_info.best_or_last

    from torch_geometric.data import Batch

    sim_batches_stacked_list = []
    for sim_batch in sim_batches:
        graph_tree = GraphTreeWrapper(sim_batch)
        sim_batches_stacked_list.append(
            batch_from_pcs_list(
                graph_tree.x_by_level[-1],
                graph_tree.batch_by_level[-1],
            )
        )

    sim_batches_stacked = Batch.from_data_list(
        [e for ee in sim_batches_stacked_list for e in ee.to_data_list()]
    )
    gen_batches_stacked = Batch.from_data_list(
        [e for ee in gen_batches for e in ee.to_data_list()]
    )

    def log_figure(figure, filename):
        outputpath = plot_path / filename
        figure.savefig(outputpath)
        figure.savefig(outputpath.with_suffix(".png"), dpi=150)
        if best_or_last == "best":
            with train_log.experiment.test():
                train_log.experiment.log_figure(
                    figure_name=filename, figure=figure, overwrite=True
                )
        logger.info(plot_path / filename)

    # Scatter of a single event
    figure = xyscatter(
        sim=sim_batches[0][0].x.numpy(),
        gen=gen_batches[0][0].x.numpy(),
        title=f"Scatter a single event ({conf.loader.max_points} points)",
    )
    log_figure(figure, "xyscatter_single.pdf")

    figure = xyscatter_faint(
        sim=sim_batches[0].x.numpy(),
        gen=gen_batches[0].x.numpy(),
        title=(
            f"Scatter points ({conf.loader.max_points}) in batch"
            f" ({conf.loader.batch_size})"
        ),
    )
    log_figure(figure, "xyscatter_batch.pdf")

    figure = xy_hist(
        sim=sim_batches_stacked.x.numpy(),
        gen=gen_batches_stacked.x.numpy(),
        title=(
            f"2D Histogram for {conf.loader.max_points} points in"
            f" {conf.testing.n_events} events"
        ),
    )
    log_figure(figure, "xy_hist.pdf")

    figure = xyscatter(
        sim=scatter_mean(
            sim_batches_stacked.x, sim_batches_stacked.batch, dim=0
        ).numpy(),
        gen=scatter_mean(
            gen_batches_stacked.x, gen_batches_stacked.batch, dim=0
        ).numpy(),
        title=f"Event means for ({conf.testing.n_events}) events",
    )
    log_figure(figure, "xy_event_means.pdf")

    for var in hlvs_dict["gen"]:
        xsim = hlvs_dict["sim"][var]
        xgen = hlvs_dict["gen"][var]
        logger.info(f"Plotting  var {var}")
        figure = hlv_marginals(var, xsim=xsim, xgen=xgen)
        log_figure(figure, f"{var}.pdf")
