import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from ..config import conf, device
from ..fw_simple_dataloader import dataset
# from ..load_sparse_ds import dataset
from ..plot import plotlosses
from ..utils.logger import logger
# from .fw_loader import costumLoader
from .holder import modelHolder

# graphlist=[dataset[i] for i in range(10)]
# train_loader = DataLoader([graphlist[1] for _ in range(10)], batch_size=4, shuffle=True)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
# for i in train_loader:
#     examplebatch=i
#     break


def training_procedure(c: modelHolder):
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=conf.model["batch_size"],
    #     shuffle=True,
    #     # num_workers=6,
    # )
    # train_loader = costumLoader(dataset)

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

            energies = torch.tensor(batch.y, dtype=torch.float32, device=device)
            c.optim.zero_grad()
            prediction = c.model(batch)

            loss = c.lossf(prediction, energies)
            loss.backward()
            c.optim.step()

            c.metrics["grad_step"] = c.metrics["grad_step"] + 1
            # save the generated torch tensor models to disk
            if c.metrics["grad_step"] % 10 == 0:
                c.metrics["loss"].append(float(loss) / ibatch)
                logger.info(
                    f"Epoch { c.metrics['epoch'] }/{conf.model['n_epochs']}: "
                    + f"\n\tLoss: {c.metrics['loss'][-1]:.1f}, "
                    + f"\n\tLast batch: {float(prediction[-1]):.1f} vs {float(energies[-1]):.1f}."
                )
            if c.metrics["grad_step"] % 50 == 0:
                c.save_model()

    plotlosses({"loss": c.metrics["loss"], "accuracy": c.metrics["acc"]})
