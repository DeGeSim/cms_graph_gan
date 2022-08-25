"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import loader_info, scaler
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.plot.xyscatter import xy_hist, xyscatter_faint

batch_size = conf.loader.batch_size


@dataclass
class TestInfo:
    train_log: TrainLog
    sim_batch: Batch
    gen_batch: Batch
    hlvs_dict: Optional[Dict[str, Dict[str, np.ndarray]]]
    plot_path: Path
    step: int
    epoch: int
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
    holder: Holder = Holder(device)
    train_log: TrainLog = holder.train_log

    test_data: TestDataset = get_testing_datasets(holder)

    for best_or_last in ["last", "best"]:
        plot_path = Path(f"{conf.path.run_path}/plots_{best_or_last}/")
        plot_path.mkdir(exist_ok=True)

        if best_or_last == "best":
            step = holder.state.best_step
            epoch = holder.state.best_epoch
            gen_batches = test_data.gen_batches_best

        else:
            step = holder.state.grad_step
            epoch = holder.state.epoch
            gen_batches = test_data.gen_batches_last

        test_info = TestInfo(
            train_log=train_log,
            sim_batch=test_data.sim_batches,
            gen_batch=gen_batches,
            hlvs_dict=test_data.hlvs_dict,
            plot_path=plot_path,
            step=step,
            epoch=epoch,
            best_or_last=best_or_last,
        )

        test_metrics(test_info)
        test_plots(test_info)

    exit(0)


def get_testing_datasets(holder: Holder) -> TestDataset:
    ds_path = Path(conf.path.run_path) / f"testdata.pt"
    test_data: TestDataset

    if ds_path.is_file():
        logger.info(f"Loading test dataset from {ds_path}")
        test_data = TestDataset(**torch.load(ds_path))
        reprocess = (
            test_data.grad_step != holder.state.grad_step
            or test_data.loader_hash != conf.loader_hash
            or test_data.hash != conf.hash
        )
    else:
        reprocess = True
    reprocess = True

    if reprocess:
        # reprocess
        # Make sure the batches are loaded
        qds: QueuedDataset = QueuedDataset(loader_info)
        # Sample at least 2k events
        # n_batches = int(conf.testing.n_events / batch_size)
        # assert n_batches <= len(loader.testing_batches)
        ds_dict = {"sim": qds.testing_batch}

        for best_or_last in ["best", "last"]:
            # Check if we need to rerun the model
            # if yes, pickle it
            if best_or_last == "best":
                holder.select_best_model()

            holder.models.eval()

            gen_graphs = []
            for _ in tqdm(range(conf.loader.test_set_size // batch_size)):
                holder.reset_gen_points()
                for igraph in range(batch_size):
                    gen_graphs.append(holder.gen_points.get_example(igraph))
            gen_batch = Batch.from_data_list(gen_graphs)

            ds_dict[best_or_last] = gen_batch

        # scale all the samples
        for k in ds_dict.keys():
            ds_dict[k].x = torch.from_numpy(
                scaler.inverse_transform(ds_dict[k].x.numpy())
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


# def compute_hlvs_dict(batches) -> Dict[str, np.ndarray]:
#     hlvs_dict_torch: Dict[str, List[torch.Tensor]] = {}
#     for batch in batches:
#         for key in batch.hlvs:
#             if key not in hlvs_dict_torch:
#                 hlvs_dict_torch[key] = []
#             hlvs_dict_torch[key].append(batch.hlvs[key])

#     # Sample 2k events and plot the distribution
#     # convert to numpy
#     hlvs_dict: Dict[str, np.ndarray] = {
#         var: subsample(convert(hlvs_dict_torch[var])) for var in hlvs_dict_torch
#     }
#     return hlvs_dict


def convert(tensorarr: List[torch.Tensor]) -> np.ndarray:
    return torch.cat(tensorarr).flatten().detach().cpu().numpy()


def subsample(arr: np.ndarray):
    return np.random.choice(arr, size=conf.testing.n_events, replace=False)


def test_metrics(test_info: TestInfo):
    train_log = test_info.train_log
    sim_batches = test_info.sim_batch
    gen_batches = test_info.gen_batch

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

    metrics_dict = {
        f"test/{test_info.best_or_last}/{k}": v for k, v in metrics_dict.items()
    }
    train_log.log_metrics(metrics_dict, step=test_info.step, epoch=test_info.epoch)


def test_plots(test_info: TestInfo):
    train_log = test_info.train_log
    sim_batch = test_info.sim_batch
    gen_batch = test_info.gen_batch
    plot_path = test_info.plot_path
    best_or_last = test_info.best_or_last

    sim_batch_small = Batch.from_data_list(sim_batch[: conf.loader.batch_size])
    gen_batch_small = Batch.from_data_list(gen_batch[: conf.loader.batch_size])

    def log_figure(figure, filename):
        outputpath = plot_path / filename
        # figure.savefig(outputpath)
        figure.savefig(outputpath.with_suffix(".png"), dpi=150)
        train_log.log_figure(
            figure_name=f"test.{best_or_last}.{filename}",
            figure=figure,
            overwrite=False,
            step=test_info.step,
        )
        logger.info(plot_path / filename)

    from itertools import combinations

    for v1, v2 in combinations(list(range(conf.loader.n_features)), 2):
        v1name = conf.loader.cell_prop_keys[v1]
        v2name = conf.loader.cell_prop_keys[v2]
        cmbname = f"{v1name}_vs_{v2name}"
        # figure = xyscatter(
        #     sim=sim_batch[0].x[:, [v1, v2]].numpy(),
        #     gen=gen_batch[0].x[:, [v1, v2]].numpy(),
        #     title=f"Scatter a single event ({conf.loader.n_points} points)",
        #     v1name=v1name,
        #     v2name=v2name,
        # )
        # log_figure(figure, f"xyscatter_single_{cmbname}.pdf")

        figure = xyscatter_faint(
            sim=sim_batch_small.x[:, [v1, v2]].numpy(),
            gen=gen_batch_small.x[:, [v1, v2]].numpy(),
            title=(
                f"Scatter points ({conf.loader.n_points}) in batch ({batch_size})"
            ),
            v1name=v1name,
            v2name=v2name,
        )
        log_figure(figure, f"xyscatter_batch_{cmbname}.pdf")

        figure = xy_hist(
            sim=sim_batch.x[:, [v1, v2]].numpy(),
            gen=gen_batch.x[:, [v1, v2]].numpy(),
            title=(
                f"2D Histogram for {conf.loader.n_points} points in"
                f" {conf.testing.n_events} events"
            ),
            v1name=v1name,
            v2name=v2name,
        )
        log_figure(figure, f"xy_hist_{cmbname}.pdf")

    from fgsim.plot.jetfeatures import jet_features

    log_figure(
        jet_features(
            sim_batch.x.reshape(
                -1, conf.loader.n_points, conf.loader.n_features
            ).numpy(),
            gen_batch.x.reshape(
                -1, conf.loader.n_points, conf.loader.n_features
            ).numpy(),
        ),
        "jetfeatures.pdf",
    )


def jetnet_metrics(sim_batch, gen_batch) -> Dict[str, float]:
    from fgsim.models.metrics import fpnd, w1efp, w1m, w1p

    metrics_dict = {}

    metrics_dict["fpnd"] = fpnd(gen_batch)
    metrics_dict["w1m"] = w1m(gen_batch=gen_batch, sim_batch=sim_batch)
    metrics_dict["w1p"] = w1p(gen_batch=gen_batch, sim_batch=sim_batch)
    metrics_dict["w1efp"] = w1efp(gen_batch=gen_batch, sim_batch=sim_batch)
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
