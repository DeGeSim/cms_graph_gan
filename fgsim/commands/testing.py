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
from fgsim.plot.validation_plots import validation_plots
from fgsim.utils.jetnetutils import to_efp

batch_size = conf.loader.batch_size


@dataclass
class TestInfo:
    train_log: TrainLog
    res_d: dict
    hlvs_dict: Optional[Dict[str, Dict[str, np.ndarray]]]
    plot_path: Path
    step: int
    epoch: int
    best_or_last: str


@dataclass
class TestDataset:
    res_d: dict
    hlvs_dict: Optional[Dict[str, Dict[str, np.ndarray]]]
    grad_step: int
    loader_hash: str
    hash: str


def test_procedure() -> None:
    holder: Holder = Holder(device)
    train_log: TrainLog = holder.train_log

    for best_or_last in ["last", "best"]:
        test_data: TestDataset = get_testing_datasets(holder, best_or_last)
        plot_path = Path(f"{conf.path.run_path}/plots_{best_or_last}/")
        plot_path.mkdir(exist_ok=True)

        if best_or_last == "best":
            step = holder.state.best_step
            epoch = holder.state.best_epoch

        else:
            step = holder.state.grad_step
            epoch = holder.state.epoch

        test_info = TestInfo(
            train_log=train_log,
            res_d=test_data.res_d,
            hlvs_dict=test_data.hlvs_dict,
            plot_path=plot_path,
            step=step,
            epoch=epoch,
            best_or_last=best_or_last,
        )

        test_metrics(test_info)

        validation_plots(
            train_log=test_info.train_log,
            res=test_data.res_d,
            plot_path=test_info.plot_path,
            best_last_val="test/" + test_info.best_or_last,
            step=test_info.step,
        )

    exit(0)


def get_testing_datasets(holder: Holder, best_or_last) -> TestDataset:
    ds_path = Path(conf.path.run_path) / f"testdata_{best_or_last}.pt"
    test_data: TestDataset

    if ds_path.is_file():
        logger.info(f"Loading test dataset from {ds_path}")
        test_data = TestDataset(
            **torch.load(ds_path, map_location=torch.device("cpu"))
        )
        reprocess = (
            test_data.grad_step != holder.state.grad_step
            or test_data.loader_hash != conf.loader_hash
            or test_data.hash != conf.hash
        )
    else:
        reprocess = True

    if reprocess:
        # reprocess
        logger.warning(f"Reprocessing {best_or_last} Dataset")
        # Make sure the batches are loaded
        qds: QueuedDataset = QueuedDataset(loader_info)
        # Sample at least 2k events
        # n_batches = int(conf.testing.n_events / batch_size)
        # assert n_batches <= len(loader.testing_batches)

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

        for test_batch in tqdm(qds.testing_batches, desc=best_or_last, miniters=20):
            for k, val in holder.pass_batch_through_model(
                test_batch.to(holder.device)
            ).items():
                if "batch" in k:
                    for e in val.to_data_list():
                        res_d_l[k].append(e.cpu())
                else:
                    res_d_l[k].append(val.cpu())

        # aggregate the results over the batches
        for k in ["sim_batch", "gen_batch"]:
            batch = Batch.from_data_list(res_d_l[k])
            batch.x = scaler.inverse_transform(batch.x)
            res_d_l[k] = batch
        for k in ["d_sim", "d_gen"]:
            res_d_l[k] = torch.hstack(res_d_l[k])

        res_d_l["sim_efps"] = to_efp(res_d_l["sim_batch"])
        res_d_l["gen_efps"] = to_efp(res_d_l["gen_batch"])

        test_data = TestDataset(
            res_d=res_d_l,
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


def test_metrics(test_info: TestInfo):
    train_log = test_info.train_log

    metrics_dict: Dict[str, float] = {}

    for k, v in jetnet_metrics(**test_info.res_d).items():
        metrics_dict[k] = v

    # KS tests
    # from scipy.stats import kstest
    # for var in hlvs_dict["best"].keys():
    #     metrics_dict[f"kstest-{var}"] = kstest(
    #         hlvs_dict["sim"][var], hlvs_dict["best"][var]
    #     ).pvalue

    train_log.log_test_metrics(
        metrics_dict,
        step=test_info.step,
        epoch=test_info.epoch,
        prefix=f"test/{test_info.best_or_last}",
    )


def jetnet_metrics(**res_d) -> Dict[str, float]:
    from fgsim.models.metrics import cov_mmd, fgd, fpnd, kpd, w1efp, w1m, w1p

    assert fpnd(res_d["sim_batch"]) < 10
    metrics_dict = {}

    metrics_dict["fpnd"] = fpnd(**res_d)
    metrics_dict["w1m"], metrics_dict["w1m_delta"] = w1m(**res_d)
    metrics_dict["w1p"], metrics_dict["w1p_delta"] = w1p(**res_d)
    metrics_dict["w1efp"], metrics_dict["w1efp_delta"] = w1efp(**res_d)
    metrics_dict["cov"], metrics_dict["mmd"] = cov_mmd(**res_d)
    metrics_dict["fgd"], metrics_dict["fgd_delta"] = fgd(**res_d)
    metrics_dict["kpd"], metrics_dict["kpd_delta"] = kpd(**res_d)

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
