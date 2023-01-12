from copy import deepcopy

import torch
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import scaler
from fgsim.ml.holder import Holder
from fgsim.ml.smoothing import smooth_features
from fgsim.monitoring.logger import logger
from fgsim.plot.validation_plots import validation_plots
from fgsim.utils.check_for_nans import check_chain_for_nans


def validate(holder: Holder, loader: QueuedDataset) -> None:
    check_chain_for_nans((holder.models,))

    # generate the batches
    res_d_l = {
        "sim_batch": [],
        "gen_batch": [],
        "d_sim": [],
        "d_gen": [],
    }
    for val_batch in loader.validation_batches:
        for k, val in holder.pass_batch_through_model(val_batch).items():
            if k in ["sim_batch", "gen_batch"]:
                for e in val.to_data_list():
                    res_d_l[k].append(e)
            else:
                res_d_l[k].append(val)
    d_sim = torch.vstack(res_d_l["d_sim"])
    d_gen = torch.vstack(res_d_l["d_gen"])

    sim_batch = Batch.from_data_list(res_d_l["sim_batch"])
    gen_batch = Batch.from_data_list(res_d_l["gen_batch"])

    max_points = conf.loader.n_points * conf.loader.validation_set_size
    assert sim_batch.x.shape[-1] == gen_batch.x.shape[-1]
    assert max_points * 0.7 <= sim_batch.x.shape[0] <= max_points
    assert max_points * 0.7 <= gen_batch.x.shape[0] <= max_points

    for batch in sim_batch, gen_batch:
        if conf.training.smoothing.active:
            batch.x = smooth_features(batch.x, holder.state.grad_step)

    results_d = {
        "sim_batch": sim_batch,
        "gen_batch": gen_batch,
        "d_gen": d_gen,
        "d_sim": d_sim,
    }

    # if holder.state.grad_step % conf.training.val.plot_interval == 0:
    #     validation_plots(
    #         train_log=holder.train_log,
    #         res=results_d,
    #         plot_path=None,
    #         best_last_val="val/scaled",
    #         step=holder.state.grad_step,
    #     )

    # scale all the samples
    for batch in results_d["sim_batch"], results_d["gen_batch"]:
        batch.x = scaler.inverse_transform(batch.x)

    if holder.state.grad_step % conf.training.val.plot_interval == 0:
        validation_plots(
            train_log=holder.train_log,
            res=results_d,
            plot_path=None,
            best_last_val="val/unscaled",
            step=holder.state.grad_step,
        )

    # evaluate the validation metrics
    with torch.no_grad():
        holder.val_metrics(**results_d)
    holder.val_metrics.log_metrics()

    # save the best model
    if max(holder.history["score"]) == holder.history["score"][-1]:
        holder.state.best_step = holder.state["grad_step"]
        holder.state.best_epoch = holder.state["epoch"]
        holder.best_model_state = deepcopy(holder.models.state_dict())
        logger.warning(f"New best model at step {holder.state.best_step}")

    holder.train_log.log_metric("other/best_step", holder.state["grad_step"])
    holder.train_log.log_metric("other/best_epoch", holder.state["epoch"])

    logger.debug("Validation done.")
