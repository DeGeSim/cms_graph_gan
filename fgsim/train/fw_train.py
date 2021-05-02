import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import conf, device
from ..fw_data_loader import dataset
from ..plot import plotlosses
from ..utils.logger import logger
from .holder import modelHolder


def training_procedure(c: modelHolder):
    # Make the configuration locally available
    batch_size, n_epochs, k, nz, sample_size = (
        conf.model.gan[x] for x in ["batch_size", "n_epochs", "k", "nz", "sample_size"]
    )

    train_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,  # num_workers=6
    )

    # Initialize the training
    c.model = c.model.float().to(device)
    c.model.train()
    logger.info(f"Starting with epoch {c.metrics['epoch'] + 1}")

    skipped_to_batch = False
    # Iterate over the Epochs
    for c.metrics["epoch"] in range(c.metrics["epoch"] + 1, n_epochs):
        # Iterate over the batches
        for ibatch, data in tqdm(enumerate(train_loader)):
            #skip to the correct batch
            if ibatch != c.metrics["batch"] and not skipped_to_batch:
                continue
            else:
                skipped_to_batch = True
                c.metrics["batch"] = ibatch

            graph_props, energies = data
            energies = torch.squeeze(energies).to(device)
            c.optim.zero_grad()
            prediction = c.model(graph_props)

            loss = c.lossf(prediction, energies)
            loss.backward()
            c.optim.step()

            c.metrics["grad_step"] = c.metrics["grad_step"] + 1
            # save the generated torch tensor models to disk
            if c.metrics["grad_step"]* % 1000 == 0:
                c.save_model()

        c.metrics["loss"].append(float(loss) / ibatch)

        logger.info(
            f"Epoch { c.metrics['epoch'] }/{n_epochs}: "
            + f"\n\tLoss: {c.metrics['loss'][-1]:.1f}, "
            + f"\n\tLast batch: {prediction[-1]:.1f} vs {energies[-1]:.1f}."
        )
    plotlosses({"loss": c.metrics["loss"], "accuracy": c.metrics["acc"]})
