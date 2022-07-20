from copy import deepcopy

import torch
from torch_geometric.data import Batch

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.utils.check_for_nans import check_chain_for_nans


def validate(holder: Holder, loader: QueuedDataset) -> None:
    holder.models.eval()
    check_chain_for_nans((holder.models,))

    with torch.no_grad():
        sim_batch = loader.validation_batch.clone().to(device)
        gen_graphs = []
        for _ in range(conf.loader.validation_set_size // conf.loader.batch_size):
            holder.reset_gen_points()
            for igraph in range(conf.loader.batch_size):
                gen_graphs.append(holder.gen_points.get_example(igraph))
        gen_batch = Batch.from_data_list(gen_graphs)

        d_sim = holder.models.disc(sim_batch)
        d_gen = holder.models.disc(gen_batch)
        holder.val_loss(
            gen_batch=gen_batch, sim_batch=sim_batch, d_gen=d_gen, d_sim=d_sim
        )
    holder.val_loss.log_metrics()

    min_stop_crit = min(holder.history["stop_crit"])
    if min_stop_crit == holder.history["stop_crit"][-1]:
        holder.state.best_grad_step = holder.state["grad_step"]
        holder.best_model_state = deepcopy(holder.models.state_dict())

        if not conf.debug:
            holder.train_log.experiment.log_metric("min_stop_crit", min_stop_crit)
            holder.train_log.experiment.log_metric(
                "best_grad_step", holder.state["grad_step"]
            )
            holder.train_log.experiment.log_metric(
                "best_epoch", holder.state["epoch"]
            )
    logger.warning("Validation function done")
    if not conf.debug:
        holder.train_log.writer.flush()
