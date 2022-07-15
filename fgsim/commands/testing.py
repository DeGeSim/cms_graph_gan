"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import jetnet
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from tqdm import tqdm

from fgsim.config import conf
from fgsim.io.batch_tools import batch_from_pcs_list
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import scaler
from fgsim.ml.holder import Holder
from fgsim.models.branching.graph_tree import graph_tree_to_graph
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.plot.xyscatter import xy_hist, xyscatter, xyscatter_faint


@dataclass
class TestInfo:
    train_log: TrainLog
    sim_batches: List
    gen_batches: List
    hlvs_dict: Optional[Dict[str, Dict[str, np.ndarray]]]
    plot_path: Path
    step: int
    best_or_last: str


@dataclass
class TestDataset:
    sim_batches: List
    gen_batches_best: List
    gen_batches_last: List
    hlvs_dict: Optional[Dict[str, Dict[str, np.ndarray]]]
    grad_step: int
    loader_hash: str
    hash: str


def test_procedure() -> None:
    holder: Holder = Holder()
    train_log: TrainLog = holder.train_log

    test_data: TestDataset = get_testing_datasets(holder)

    for best_or_last in ["last"]:
        plot_path = Path(f"{conf.path.run_path}/plots_{best_or_last}/")
        plot_path.mkdir(exist_ok=True)

        if best_or_last == "best":
            step = holder.state.best_grad_step
            gen_batches = test_data.gen_batches_best

        else:
            step = holder.state.grad_step
            gen_batches = test_data.gen_batches_last

        test_info = TestInfo(
            train_log=train_log,
            sim_batches=test_data.sim_batches,
            gen_batches=gen_batches,
            hlvs_dict=test_data.hlvs_dict,
            plot_path=plot_path,
            step=step,
            best_or_last=best_or_last,
        )

        test_metrics(test_info)
        test_plots(test_info)

    exit(0)


def get_testing_datasets(holder: Holder) -> TestDataset:
    ds_path = Path(conf.path.run_path) / f"testdata.pt"
    test_data: TestDataset
    reprocess = not ds_path.is_file()
    if not reprocess:
        logger.info(f"Loading test dataset from {ds_path}")
        test_data = TestDataset(**torch.load(ds_path))
        reprocess = (
            test_data.grad_step != holder.state.grad_step
            or test_data.loader_hash != conf.loader_hash
            or test_data.hash != conf.hash
        )

    if reprocess:
        # reprocess
        # Make sure the batches are loaded
        loader: QueuedDataset = QueuedDataset()
        # Sample at least 2k events
        # n_batches = int(conf.testing.n_events / conf.loader.batch_size)
        # assert n_batches <= len(loader.testing_batches)
        ds_dict = {"sim": loader.testing_batches}

        for best_or_last in ["best", "last"]:
            # Check if we need to rerun the model
            # if yes, pickle it
            if best_or_last == "best":
                holder.select_best_model()

            holder.models.eval()

            gen_batches = []
            # generate a batch for each simulated one:
            for _ in tqdm(
                range(len(loader.testing_batches)),
                desc=f"Generating Batches {best_or_last}",
            ):
                with torch.no_grad():
                    holder.reset_gen_points()
                    holder.gen_points = graph_tree_to_graph(holder.gen_points)
                    gen_batch = holder.gen_points.clone().cpu()
                    gen_batches.append(gen_batch)

            logger.info("Processs HLVs")
            ds_dict[best_or_last] = gen_batches
            # from fgsim.io.batch_tools import batch_compute_hlvs
            # ds_dict[best_or_last] = [
            #     batch_compute_hlvs(batch)
            #     for batch in tqdm(gen_batches, desc="Compute HLVs gen_batches")
            # ]
        # hlvs_dict: Dict[str, Dict[str, np.ndarray]] = {
        #     name: compute_hlvs_dict(ds) for name, ds in ds_dict.items()
        # }
        # scale all the samples
        for k in ds_dict.keys():
            for ibatch in tqdm(range(len(ds_dict[k])), desc=f"Scaling {k}"):
                batch = ds_dict[k][ibatch]
                batch.x = torch.from_numpy(
                    scaler.inverse_transform(batch.x.numpy())
                )
        test_data = TestDataset(
            sim_batches=ds_dict["sim"],
            gen_batches_best=ds_dict["best"],
            gen_batches_last=ds_dict["last"],
            grad_step=holder.state.grad_step,
            hlvs_dict=None,
            loader_hash=conf.loader_hash,
            hash=conf.hash,
        )
        logger.info(f"Saving test dataset to {ds_path}")
        torch.save(
            test_data.__dict__,
            ds_path,
        )

    return test_data


def compute_hlvs_dict(batches) -> Dict[str, np.ndarray]:
    hlvs_dict_torch: Dict[str, List[torch.Tensor]] = {}
    for batch in batches:
        for key in batch.hlvs:
            if key not in hlvs_dict_torch:
                hlvs_dict_torch[key] = []
            hlvs_dict_torch[key].append(batch.hlvs[key])

    # Sample 2k events and plot the distribution
    # convert to numpy
    hlvs_dict: Dict[str, np.ndarray] = {
        var: subsample(convert(hlvs_dict_torch[var])) for var in hlvs_dict_torch
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

    metrics_dict: Dict[str, float] = {}

    # for k, v in w_metrics(sim_batches, gen_batches).items():
    #     metrics_dict[k] = v
    for k, v in jetnet_metrics(sim_batches, gen_batches).items():
        metrics_dict[k] = v

    # KS tests
    # from scipy.stats import kstest
    # for var in hlvs_dict["best"].keys():
    #     metrics_dict[f"kstest-{var}"] = kstest(
    #         hlvs_dict["sim"][var], hlvs_dict["best"][var]
    #     ).pvalue

    metrics_dict = {f"test.{k}": v for k, v in metrics_dict.items()}
    train_log.experiment.log_metrics(
        metrics_dict,
        step=test_info.step,
    )


def test_plots(test_info: TestInfo):
    train_log = test_info.train_log
    sim_batches = test_info.sim_batches
    gen_batches = test_info.gen_batches
    # hlvs_dict = test_info.hlvs_dict
    plot_path = test_info.plot_path
    best_or_last = test_info.best_or_last

    sim_batches_stacked_list = []
    for sim_batch in sim_batches:
        sim_batches_stacked_list.append(
            batch_from_pcs_list(
                sim_batch.x,
                sim_batch.batch,
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
        # figure.savefig(outputpath)
        figure.savefig(outputpath.with_suffix(".png"), dpi=150)
        train_log.experiment.log_figure(
            figure_name=f"test.{best_or_last}.{filename}",
            figure=figure,
            overwrite=True,
        )
        logger.info(plot_path / filename)

    # Scatter of a single event
    from itertools import combinations

    for v1, v2 in combinations(list(range(conf.loader.n_features)), 2):
        v1name = conf.loader.cell_prop_keys[v1]
        v2name = conf.loader.cell_prop_keys[v2]
        cmbname = f"{v1name}_vs_{v2name}"
        figure = xyscatter(
            sim=sim_batches[0][0].x[:, [v1, v2]].numpy(),
            gen=gen_batches[0][0].x[:, [v1, v2]].numpy(),
            title=f"Scatter a single event ({conf.loader.max_points} points)",
            v1name=v1name,
            v2name=v2name,
        )
        log_figure(figure, f"xyscatter_single_{cmbname}.pdf")

        figure = xyscatter_faint(
            sim=sim_batches[0].x[:, [v1, v2]].numpy(),
            gen=gen_batches[0].x[:, [v1, v2]].numpy(),
            title=(
                f"Scatter points ({conf.loader.max_points}) in batch"
                f" ({conf.loader.batch_size})"
            ),
            v1name=v1name,
            v2name=v2name,
        )
        log_figure(figure, f"xyscatter_batch_{cmbname}.pdf")

        figure = xy_hist(
            sim=sim_batches_stacked.x[:, [v1, v2]].numpy(),
            gen=gen_batches_stacked.x[:, [v1, v2]].numpy(),
            title=(
                f"2D Histogram for {conf.loader.max_points} points in"
                f" {conf.testing.n_events} events"
            ),
            v1name=v1name,
            v2name=v2name,
        )
        log_figure(figure, f"xy_hist_{cmbname}.pdf")


def jetnet_metrics(sim_batches, gen_batches) -> Dict[str, float]:
    sim = np.hstack(
        [
            sim_batches[0].x.numpy(),
            sim_batches[0].mask.reshape(-1, 1).float().numpy(),
        ],
    )
    sim[~sim[:, -1].astype(dtype=bool), :3] = 0
    sim = sim.reshape(conf.loader.batch_size, 30, 4)

    gen = np.hstack(
        [
            gen_batches[0].x.numpy(),
            torch.ones(conf.loader.batch_size * 30, 1).float().numpy(),
        ],
    )
    gen[~gen[:, -1].astype(dtype=bool), :3] = 0
    gen = gen.reshape(conf.loader.batch_size, 30, 4)
    metrics_dict = {}
    metrics_dict["fpnd"] = jetnet.evaluation.fpnd(
        jets=gen[..., :3], jet_type="t", batch_size=conf.loader.batch_size
    )

    metrics_dict["w1m"] = (
        jetnet.evaluation.w1m(
            jets1=sim[..., :3],
            jets2=gen[..., :3],
        )[0]
        * 1e3
    )

    metrics_dict["w1p"] = (
        jetnet.evaluation.w1p(
            jets1=sim[..., :3],
            jets2=gen[..., :3],
            # mask1=sim[..., 3].squeeze(),
            # mask2=gen[..., 3].squeeze(),
        )[0]
        * 1e3
    )

    metrics_dict["w1efp"] = (
        jetnet.evaluation.w1efp(
            jets1=sim[..., :3],
            jets2=gen[..., :3],
        )[0]
        * 1e5
    )
    return metrics_dict


def w_metrics(sim_batches, gen_batches) -> Dict[str, float]:
    sim = sim_batches[0].x.numpy()
    gen = gen_batches[0].x.numpy()

    sim_means = scatter_mean(sim_batches[0].x, sim_batches[0].batch, dim=0).numpy()
    gen_means = scatter_mean(gen_batches[0].x, gen_batches[0].batch, dim=0).numpy()

    # compute a covar matrix for each batch
    # take the sqrt of the elements to scale to the scale of the variable
    # and then compare the distributions with w1

    covars_sim = torch.vstack(
        [torch.cov(batch.x[:, :2].T).reshape(1, 4) for batch in sim_batches]
    ).numpy()
    covars_gen = torch.vstack(
        [torch.cov(batch.x[:, :2].T).reshape(1, 4) for batch in gen_batches]
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
    return metrics_dict
