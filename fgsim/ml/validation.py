from copy import deepcopy

import torch
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.ml.holder import Holder
from fgsim.monitoring.logger import logger
from fgsim.utils.check_for_nans import check_chain_for_nans


def validate(holder: Holder, loader: QueuedDataset) -> None:
    holder.models.eval()
    check_chain_for_nans((holder.models,))

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

        holder.val_loss(
            gen_batch=gen_batch, sim_batch=sim_batch, d_gen=d_gen, d_sim=d_sim
        )
    holder.val_loss.log_metrics()

    min_stop_crit = min(holder.history["stop_crit"])
    if min_stop_crit == holder.history["stop_crit"][-1]:
        holder.state.best_step = holder.state["grad_step"]
        holder.state.best_epoch = holder.state["epoch"]
        holder.best_model_state = deepcopy(holder.models.state_dict())

        if not conf.debug:
            holder.train_log.log_metric("min_stop_crit", min_stop_crit)
            holder.train_log.log_metric("best_step", holder.state["grad_step"])
            holder.train_log.log_metric("best_epoch", holder.state["epoch"])
    if not conf.debug:
        holder.train_log.writer.flush()
    logger.debug("Validation done.")
