from copy import deepcopy

import torch
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import scaler
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.plot.validation_plots import validation_plots
from fgsim.utils.check_for_nans import check_chain_for_nans


def validate(holder: Holder, loader: QueuedDataset) -> None:
    holder.models.eval()
    check_chain_for_nans((holder.models,))

    # generate the batches
    with torch.no_grad():
        gen_graphs = []
        for _ in range(len(loader.validation_batches)):
            holder.reset_gen_points()
            for igraph in range(conf.loader.batch_size):
                gen_graphs.append(holder.gen_points.get_example(igraph))
        d_sim = torch.hstack(
            [holder.models.disc(batch) for batch in loader.validation_batches]
        )
        d_gen = torch.hstack([holder.models.disc(batch) for batch in gen_graphs])

        sim_batch = Batch.from_data_list(loader.validation_batches)
        gen_batch = Batch.from_data_list(gen_graphs)

    # scale all the samples
    for batch in sim_batch, gen_batch:
        batch.x = scaler.inverse_transform(batch.x)

    # evaluate the validation metrics
    with torch.no_grad():
        holder.val_loss(
            gen_batch=gen_batch, sim_batch=sim_batch, d_gen=d_gen, d_sim=d_sim
        )
    holder.val_loss.log_metrics()

    # validation plots
    validation_plots(
        train_log=holder.train_log,
        sim_batch=sim_batch,
        gen_batch=gen_batch,
        plot_path=None,
        best_last_val="val",
        step=holder.state.grad_step,
    )

    # select the best model
    min_stop_crit = min(holder.history["stop_crit"])
    if min_stop_crit == holder.history["stop_crit"][-1]:
        holder.state.best_step = holder.state["grad_step"]
        holder.state.best_epoch = holder.state["epoch"]
        holder.best_model_state = deepcopy(holder.models.state_dict())

        holder.train_log.log_metric("other/min_stop_crit", min_stop_crit)
        holder.train_log.log_metric("other/best_step", holder.state["grad_step"])
        holder.train_log.log_metric("other/best_epoch", holder.state["epoch"])

    logger.debug("Validation done.")
