import torch
import time
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

from tqdm import tqdm

from ..config import conf, device
from ..io.dataset import dataset
from ..utils.logger import logger
from .holder import modelHolder


def diffperc(a, b):
    return torch.abs(torch.mean((a - b) / b)) * 100


def training_procedure(c: modelHolder):
    train_loader = DataLoader(dataset, batch_size=conf.model.batch_size, shuffle=True)

    # Initialize the training
    c.model = c.model.float().to(device)
    c.model.train()
    logger.info(f"Starting with epoch {c.metrics['epoch']}")

    skipped_to_batch = False
    # Iterate over the Epochs
    for c.metrics["epoch"] in range(c.metrics["epoch"], conf.model["n_epochs"]):
        # Iterate over the batches
        for ibatch, batch in tqdm(enumerate(train_loader)):
            # skip to the correct batch
            # This construction allows restarting the training during an epoch.
            if ibatch != c.metrics["batch"] and not skipped_to_batch:
                continue
            else:
                skipped_to_batch = True
                c.metrics["batch"] = ibatch

            c.optim.zero_grad()
            prediction = torch.squeeze(c.model(batch).T)

            loss = c.lossf(prediction, batch.y.float())
            loss.backward()
            c.optim.step()

            c.metrics["grad_step"] = c.metrics["grad_step"] + 1
            writer.add_scalar("loss", loss, c.metrics["grad_step"])

            nndiff = diffperc(prediction, batch.y.float())
            writer.add_scalar("nndiff", nndiff, c.metrics["grad_step"])

            simple_sum = global_add_pool(batch.x, batch.batch)
            simplediff = diffperc(simple_sum, batch.y.float())
            writer.add_scalar("simplediff", simplediff, c.metrics["grad_step"])

            writer.add_scalar(
                "nndiff - simplediff", nndiff - simplediff, c.metrics["grad_step"]
            )

            writer.flush()

            # save the generated torch tensor models to disk
            if c.metrics["grad_step"] % 10 == 0:
                logger.info(
                    f"Batch {ibatch}/{len(train_loader)} "
                    + f"Epoch { c.metrics['epoch'] }/{conf.model['n_epochs']}: "
                    + f"\n\tLoss: {nndiff}"
                )
            if c.metrics["grad_step"] % 50 == 0:
                c.save_model()
        c.save_model()
        break