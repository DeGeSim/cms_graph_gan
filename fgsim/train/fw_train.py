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

    # Iterate over the Epochs
    for c.metrics["epoch"] in range(c.metrics["epoch"] + 1, n_epochs):
        # n_iter = int(len(train_data) / batch_size)

        # Iterate over the batches
        # for local_batch, local_labels in train_loader:
        # with (0, next(dataset)) as ibatch, data:
        for ibatch, data in tqdm(enumerate(train_loader)):
            graph_props, energies = data
            # logger.info("Training - Got to the inner loop")
            c.optim.zero_grad()
            out = c.model(graph_props)

            loss = c.lossf(out, energies.to(device))
            loss.backward()
            c.optim.step()

        # save the generated torch tensor models to disk
        if c.metrics["epoch"] % 10 == 0:
            c.save_model()

        c.metrics["losses"].append(float(loss) / ibatch)

        logger.info(
            f"Epoch { c.metrics['epoch'] }/{n_epochs}: "
            + f"Generator loss: {c.metrics['losses'][-1]:.8f}, "
            + f"Last batch: {out[-1]} vs {energies[-1]}."
        )
    plotlosses(c.metrics["losses"], c.metrics["losses_d"])
