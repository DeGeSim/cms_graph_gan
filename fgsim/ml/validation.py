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

    # evaluate the validation losses
    with torch.no_grad():
        holder.val_loss(
            gen_batch=gen_batch, sim_batch=sim_batch, d_gen=d_gen, d_sim=d_sim
        )
    holder.val_loss.log_metrics()

    # validation metrics

    # validation plots
    logger.debug("Validation done.")
