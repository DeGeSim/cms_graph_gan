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

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import loader_info, scaler
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.plot.validation_plots import validation_plots

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

        validation_plots(
            train_log=test_info.train_log,
            sim_batch=test_info.sim_batch,
            gen_batch=test_info.gen_batch,
            plot_path=test_info.plot_path,
            best_last_val="test/" + test_info.best_or_last,
            step=test_info.step,
        )

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

    if reprocess:
        # reprocess
        # Make sure the batches are loaded
        qds: QueuedDataset = QueuedDataset(loader_info)
        # Sample at least 2k events
        # n_batches = int(conf.testing.n_events / batch_size)
        # assert n_batches <= len(loader.testing_batches)
        ds_dict: Dict[str, Dict[str, Batch]] = {}

        for best_or_last in ["best", "last"]:
            # Check if we need to rerun the model
            # if yes, pickle it
            if best_or_last == "best":
                holder.select_best_model()

            holder.models.eval()

            res_d_l = {
                "sim_batch": [],
                "gen_batch": [],
                "d_sim": [],
                "d_gen": [],
            }
            for test_batch in qds.testing_batches:
                for k, val in holder.pass_batch_through_model(
                    test_batch.to(holder.device)
                ).items():
                    if "batch" in k:
                        for e in val.to_data_list():
                            res_d_l[k].append(e)
                    else:
                        res_d_l[k].append(val)
            # d_sim = torch.hstack(res_d_l["d_sim"])
            # d_gen = torch.hstack(res_d_l["d_gen"])
            if "sim" not in ds_dict:
                ds_dict["sim"] = Batch.from_data_list(res_d_l["sim_batch"]).cpu()
            ds_dict[best_or_last] = Batch.from_data_list(res_d_l["gen_batch"]).cpu()

        # scale all the samples
        for k in ds_dict.keys():
            if isinstance(ds_dict[k], Batch):
                ds_dict[k].x = scaler.inverse_transform(ds_dict[k].x)
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


def convert(tensorarr: List[torch.Tensor]) -> np.ndarray:
    return torch.cat(tensorarr).flatten().detach().cpu().numpy()


def subsample(arr: np.ndarray):
    return np.random.choice(arr, size=conf.testing.n_events, replace=False)


def test_metrics(test_info: TestInfo):
    train_log = test_info.train_log
    sim_batches = test_info.sim_batch
    gen_batches = test_info.gen_batch

    metrics_dict: Dict[str, float] = {}

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


def jetnet_metrics(sim_batch, gen_batch) -> Dict[str, float]:
    from fgsim.models.metrics import fpnd, w1efp, w1m, w1p

    assert fpnd(sim_batch) < 10
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
