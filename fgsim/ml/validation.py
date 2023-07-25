from copy import deepcopy

from fgsim.io.queued_dataset import QueuedDataset
from fgsim.ml.holder import Holder
from fgsim.monitoring import logger
from fgsim.utils.check_for_nans import check_chain_for_nans

from .eval import eval_res_d, gen_res_from_sim_batches


def validate(holder: Holder, loader: QueuedDataset) -> None:
    check_chain_for_nans((holder.models,))

    # generate the batches
    logger.debug("Val: Running Generator and Critic")
    results_d = gen_res_from_sim_batches(loader.validation_batches, holder)
    logger.debug("Validation batches created")

    eval_res_d(
        results_d,
        holder,
        holder.state["grad_step"],
        holder.state["epoch"],
        ["val"],
        None,
    )

    # # overwrite the recorded score for each val step
    # import wandb
    # for ivalstep in range(len(score)):
    #     scoreistep = ivalstep * conf.training.val.interval
    #     wandb.log(
    #         {"trend/score": score[ivalstep]},
    #         step=scoreistep,
    #         epoch=scoreistep // loader.n_grad_steps_per_epoch,
    #     )

    # save the best model
    if max(holder.history["score"]) == holder.history["score"][-1]:
        holder.state.best_step = holder.state["grad_step"]
        holder.state.best_epoch = holder.state["epoch"]
        holder.best_model_state = deepcopy(holder.models.state_dict())
        if len(holder.swa_models):
            holder.best_swa_model_state = {
                n: deepcopy(p.state_dict()) for n, p in holder.swa_models.items()
            }
        logger.warning(f"New best model at step {holder.state.best_step}")

    step = holder.state.grad_step
    epoch = holder.state.epoch
    holder.train_log.log_metrics(
        {
            "best_step": holder.state["best_step"],
            "best_epoch": holder.state["best_epoch"],
        },
        prefix="trend",
        step=step,
        epoch=epoch,
    )

    logger.debug("Validation done.")
