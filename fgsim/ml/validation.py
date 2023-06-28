from copy import deepcopy

import torch
from torch_geometric.data import Batch

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import scaler
from fgsim.ml.holder import Holder
from fgsim.ml.smoothing import smooth_features
from fgsim.monitoring import logger
from fgsim.plot.fig_logger import FigLogger
from fgsim.plot.modelgrads import fig_grads
from fgsim.plot.validation_plots import validation_plots
from fgsim.utils.check_for_nans import check_chain_for_nans


def validate(holder: Holder, loader: QueuedDataset) -> None:
    check_chain_for_nans((holder.models,))
    step = holder.state.grad_step

    # generate the batches
    logger.debug("Val: Running Generator and Critic")
    res_d_l = {
        "sim_batch": [],
        "gen_batch": [],
        "sim_crit": [],
        "gen_crit": [],
    }
    for val_batch in loader.validation_batches:
        val_batch = val_batch.to(device)
        for k, val in holder.pass_batch_through_model(val_batch, eval=True).items():
            if k in ["sim_batch", "gen_batch"]:
                for e in val.to_data_list():
                    res_d_l[k].append(e.detach().cpu())
            elif k in ["sim_crit", "gen_crit"]:
                res_d_l[k].append(val.detach().cpu())
    logger.debug("Validation batches created")
    sim_crit = torch.vstack(res_d_l["sim_crit"])
    gen_crit = torch.vstack(res_d_l["gen_crit"])

    sim_batch = Batch.from_data_list(res_d_l["sim_batch"])
    gen_batch = Batch.from_data_list(res_d_l["gen_batch"])

    # max_points = conf.loader.n_points * conf.loader.validation_set_size
    assert sim_batch.x.shape == gen_batch.x.shape

    for batch in sim_batch, gen_batch:
        if conf.training.smoothing.active:
            batch.x = smooth_features(batch.x, step)

    results_d = {
        "sim_batch": sim_batch,
        "gen_batch": gen_batch,
        "gen_crit": gen_crit,
        "sim_crit": sim_crit,
    }

    # if step % conf.training.val.plot_interval == 0:
    #     validation_plots(
    #         train_log=holder.train_log,
    #         res=results_d,
    #         plot_path=None,
    #         best_last_val="val/scaled",
    #         step=step,
    #     )

    logger.debug("Start scaling")
    # scale all the samples
    for batch in results_d["sim_batch"], results_d["gen_batch"]:
        batch.x = scaler.inverse_transform(batch.x, "x")
    results_d["sim_batch"].y = scaler.inverse_transform(
        results_d["sim_batch"].y, "y"
    )
    logger.debug("End scaling")

    if not conf.debug:
        if step % conf.training.val.plot_interval == 0:
            best_last_val = "val/unscaled"
            fig_logger = FigLogger(
                holder.train_log,
                plot_path=None,
                best_last_val=best_last_val,
                step=step if step != 0 else 1,
            )
            validation_plots(
                fig_logger=fig_logger, res=results_d, best_last_val=best_last_val
            )
            for lpart in holder.losses:
                if len(lpart.grad_aggr.steps):
                    grad_fig = fig_grads(lpart.grad_aggr, lpart.name)
                    fig_logger(grad_fig, f"grads/{lpart.name}")

    if len({"kpd", "fgd"} & set(conf.training.val.metrics)):
        from fgsim.utils.jetnetutils import to_efp

        results_d["sim_efps"] = to_efp(results_d["sim_batch"])
        results_d["gen_efps"] = to_efp(results_d["gen_batch"])

    if conf.dataset_name == "calochallange":
        from fgsim.loaders.calochallange.convcoord import batch_to_Exyz

        results_d["sim_batch"] = batch_to_Exyz(results_d["sim_batch"])
        results_d["gen_batch"] = batch_to_Exyz(results_d["gen_batch"])

    # evaluate the validation metrics
    with torch.no_grad():
        holder.val_metrics(**results_d)
    holder.val_metrics.log_metrics(loader.n_grad_steps_per_epoch)
    if conf.debug:
        return
    # save the best model
    if max(holder.history["score"]) == holder.history["score"][-1]:
        holder.state.best_step = holder.state["grad_step"]
        holder.state.best_epoch = holder.state["epoch"]
        holder.best_model_state = deepcopy(holder.models.state_dict())
        logger.warning(f"New best model at step {holder.state.best_step}")

    holder.train_log.log_metrics(
        {
            "best_step": holder.state["best_step"],
            "best_epoch": holder.state["best_epoch"],
        },
        prefix="trend",
        commit=True,
    )

    logger.debug("Validation done.")
