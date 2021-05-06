from multiprocessing import Pool

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import conf, device

# from ..fw_data_loader import dataset
from ..fw_simple_dataloader import dataset
from ..plot import plotlosses
from ..utils.logger import logger
from .holder import modelHolder

# Define and keep a Pool to load the Datafrom the dataset
p = Pool(10)
def costumLoader(ds):
    idxs = np.arange(len(ds))
    np.random.shuffle(idxs)
    batch_size = conf.model["batch_size"]
    batched = np.array(
        [
            idxs[i * batch_size: (i + 1) * batch_size]
            for i in range(len(ds) // batch_size + (1 if len(ds) % batch_size else 0))
        ],dtype=object
    )
    for batch in batched:
        foo = p.map(ds.__getitem__, batch)
        yield foo


def training_procedure(c: modelHolder):
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=conf.model["batch_size"],
    #     shuffle=True,
    #     # num_workers=6,
    # )
    train_loader = costumLoader(dataset)

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

            graph_props = [e[0] for e in batch]
            energies = [e[1] for e in batch]
            energies = torch.tensor(energies).to(device)
            c.optim.zero_grad()
            prediction = c.model(graph_props)

            loss = c.lossf(prediction, energies)
            loss.backward()
            c.optim.step()

            c.metrics["grad_step"] = c.metrics["grad_step"] + 1
            # save the generated torch tensor models to disk
            if c.metrics["grad_step"] % 10 == 0:
                logger.info(    
                    f"Epoch { c.metrics['epoch'] }/{conf.model['n_epochs']}: "
                    + f"\n\tLoss: {c.metrics['loss'][-1]:.1f}, "
                    + f"\n\tLast batch: {prediction[-1]:.1f} vs {energies[-1]:.1f}."
                )
            if c.metrics["grad_step"] % 50 == 0:
                c.save_model()
        c.metrics["loss"].append(float(loss) / ibatch)


    plotlosses({"loss": c.metrics["loss"], "accuracy": c.metrics["acc"]})
