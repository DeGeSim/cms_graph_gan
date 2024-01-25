from datetime import datetime, timedelta

import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.datasets import postprocess as dataset_postprocess
from fgsim.datasets import scaler
from fgsim.ml.holder import Holder
from fgsim.plot.eval_plots import eval_plots
from fgsim.plot.fig_logger import FigLogger
from fgsim.plot.modelgrads import fig_grads


class catchtime:
    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, type, value, traceback):
        self.delta = datetime.now() - self.start
        # self.readout = f'Time: {self.time:.3f} seconds'


def gen_res_from_sim_batches(batches: list[Batch], holder: Holder):
    res_d_l = {
        "sim_batch": [],
        "gen_batch": [],
        "sim_crit": [],
        "gen_crit": [],
    }
    gen_time = timedelta(0)
    crit_time = timedelta(0)

    for batch in tqdm(
        batches, "Generating eval batches", miniters=20, mininterval=5.0
    ):
        batch = batch.to(device)
        res = {"sim_batch": batch}
        with catchtime() as td:
            res = holder.pass_batch_through_gen(res, eval=True)
        gen_time += td.delta

        with catchtime() as td:
            res = holder.pass_batch_through_crit(res, eval=True)
        crit_time += td.delta

        for k, val in res.items():
            if k in ["sim_batch", "gen_batch"]:
                for e in val.to_data_list():
                    res_d_l[k].append(e)
            elif k in ["sim_crit", "gen_crit"]:
                res_d_l[k].append(val)

    if conf.command == "test":
        holder.train_log.log_summary(
            {
                "event_gen_time": gen_time
                / timedelta(milliseconds=1)
                / conf.loader.test_set_size,
                "event_crit_time": crit_time
                / timedelta(milliseconds=1)
                / conf.loader.test_set_size,
            },
            prefix="speed",
        )

    with catchtime() as td:
        sim_crit = torch.vstack(res_d_l["sim_crit"])
        gen_crit = torch.vstack(res_d_l["gen_crit"])

        sim_batch = Batch.from_data_list(res_d_l["sim_batch"])
        gen_batch = Batch.from_data_list(res_d_l["gen_batch"])

        del res_d_l
        assert sim_batch.x.shape == gen_batch.x.shape

        gen_batch.y = sim_batch.y.clone()
        sim_batch = postprocess(sim_batch, "sim")
        gen_batch = postprocess(gen_batch, "gen")

    if conf.command == "test":
        holder.train_log.log_summary(
            {
                "event_pp_time": td.delta
                / timedelta(milliseconds=1)
                / conf.loader.test_set_size,
            },
            prefix="speed",
        )

    if conf.dataset_name == "calochallange":
        assert gen_batch.x.shape <= sim_batch.x.shape
        # assert (
        #     sim_batch.n_pointsv == gen_batch.n_pointsv + gen_batch.n_multihit
        # ).all()
    else:
        assert gen_batch.x.shape == sim_batch.x.shape

    results_d = {
        "sim_batch": sim_batch,
        "gen_batch": gen_batch,
        "gen_crit": gen_crit,
        "sim_crit": sim_crit,
    }
    return results_d


def postprocess(batch: Batch, sim_or_gen: str) -> Batch:
    batch.x_scaled = batch.x.clone()
    batch.x = scaler.inverse_transform(batch.x, "x")
    if "y" in batch.keys:
        batch.y_scaled = batch.y.clone()
        batch.y = scaler.inverse_transform(batch.y, "y")

    batch = dataset_postprocess(batch, sim_or_gen)
    return batch


def eval_res_d(
    results_d: dict,
    holder: Holder,
    step: int,
    epoch: int,
    mode: list[str],
    plot_path=None,
):
    plot = step % conf.training.plot_interval == 0 or conf.command == "test"
    if conf.command == "test":
        plot = False

    # evaluate the validation metrics
    with torch.no_grad():
        holder.eval_metrics(**results_d)
    up_metrics_d, score = holder.eval_metrics.get_metrics()

    if conf.command != "test":
        holder.train_log.log_metrics(
            up_metrics_d, prefix="/".join(mode), step=step, epoch=epoch
        )
    else:
        holder.train_log.log_summary(up_metrics_d, prefix="/".join(mode))

    if not plot:
        return score

    fig_logger = FigLogger(
        holder.train_log,
        plot_path=plot_path,
        prefixes=mode,
        step=step,
        epoch=epoch,
    )

    eval_plots(fig_logger=fig_logger, res=results_d)

    if mode[0] == "test":
        return score

    # Plot the gradients and the weights
    for lpart in holder.losses:
        if len(lpart.grad_aggr.steps):
            grad_fig = fig_grads(lpart.grad_aggr, lpart.name)
            fig_logger(grad_fig, f"grads/{lpart.name}")

    return score
