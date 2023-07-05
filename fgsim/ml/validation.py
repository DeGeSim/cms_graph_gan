from copy import deepcopy

import torch
from torch_geometric.data import Batch

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.ml.holder import Holder
from fgsim.monitoring import logger
from fgsim.utils.check_for_nans import check_chain_for_nans

from .eval import eval_res_d, postprocess


def validate(holder: Holder, loader: QueuedDataset) -> None:
    check_chain_for_nans((holder.models,))

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

    assert sim_batch.x.shape == gen_batch.x.shape

    results_d = {
        "sim_batch": sim_batch,
        "gen_batch": gen_batch,
        "gen_crit": gen_crit,
        "sim_crit": sim_crit,
    }

    for k in ["sim_batch", "gen_batch"]:
        results_d[k] = postprocess(results_d[k])

    score = eval_res_d(results_d, holder)

    # overwrite the recorded score for each val step

    for ivalstep in range(len(score)):
        scoreistep = ivalstep * conf.training.val.interval

        holder.train_log.log_metrics(
            {"trend/score": score[ivalstep]},
            step=scoreistep,
            epoch=scoreistep // loader.n_grad_steps_per_epoch,
        )

    # save the best model
    if max(holder.history["score"]) == holder.history["score"][-1]:
        holder.state.best_step = holder.state["grad_step"]
        holder.state.best_epoch = holder.state["epoch"]
        holder.best_model_state = deepcopy(holder.models.state_dict())
        logger.warning(f"New best model at step {holder.state.best_step}")

    step = holder.state.grad_step
    step = step if step != 0 else 1
    epoch = holder.state.epoch
    holder.train_log.log_metrics(
        {
            "best_step": holder.state["best_step"],
            "best_epoch": holder.state["best_epoch"],
        },
        prefix="trend",
        commit=True,
        step=step,
        epoch=epoch,
    )

    logger.debug("Validation done.")
