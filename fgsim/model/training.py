import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from ..config import conf, device
from ..io.dataset import dataset
from ..utils.logger import logger
from .holder import modelHolder



def training_procedure(c: modelHolder):
    train_loader = DataLoader(dataset, batch_size=100, shuffle=True)

    # Initialize the training
    c.model = c.model.float().to(device)
    c.model.train()
    logger.info(f"Starting with epoch {c.metrics['epoch'] + 1}")

    skipped_to_batch = False
    # Iterate over the Epochs
    for c.metrics["epoch"] in range(c.metrics["epoch"] + 1, conf.model["n_epochs"]):
        # Iterate over the batches
        for ibatch, batch in tqdm(enumerate(train_loader)):
            # skip to the correct batch
            if ibatch != c.metrics["batch"] and not skipped_to_batch:
                continue
            else:
                skipped_to_batch = True
                c.metrics["batch"] = ibatch

            c.optim.zero_grad()
            prediction = torch.squeeze( c.model(batch).T) 

            loss = c.lossf(prediction, batch.y.float())
            loss.backward()
            c.optim.step()

            c.metrics["grad_step"] = c.metrics["grad_step"] + 1
            # save the generated torch tensor models to disk
            if c.metrics["grad_step"] % 10 == 0:
                c.metrics["loss"].append(float(loss) / ibatch)
                logger.info(
                    f"Epoch { c.metrics['epoch'] }/{conf.model['n_epochs']}: "
                    + f"\n\tLoss: {c.metrics['loss'][-1]:.1f}, "
                    + f"\n\tLast batch: {float(prediction[-1]):.1f} vs {float(batch.y[-1]):.1f}."
                )
            if c.metrics["grad_step"] % 50 == 0:
                c.save_model()