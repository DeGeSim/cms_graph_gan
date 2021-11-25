"""Contain the training procedure, access point from __main__"""

import time

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import BatchType, QueuedDataLoader
from fgsim.ml.early_stopping import early_stopping
from fgsim.ml.holder import Holder
from fgsim.ml.validate import validate
from fgsim.models.loss.gradient_penalty import GradientPenalty
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.utils.batch_utils import move_batch_to_device

GP = GradientPenalty(10, gamma=1, device=device)


def training_step(
    batch: BatchType,
    holder: Holder,
) -> None:
    # set all optimizers to a 0 gradient
    holder.optims.zero_grad()

    # discriminator
    z = torch.randn(conf.loader.batch_size, 1, 96).to(device)
    tree = [z]

    with torch.no_grad():
        fake_point = holder.models.gen(tree)

    D_real = holder.models.disc(batch)
    D_realm = D_real.mean()

    D_fake = holder.models.disc(fake_point)
    D_fakem = D_fake.mean()

    gp_loss = GP(holder.models.disc, batch.data, fake_point.data)

    d_loss = -D_realm + D_fakem
    d_loss_gp = d_loss + gp_loss
    d_loss_gp.backward()
    holder.optims.disc.step()

    # generator
    if holder.state.ibatch % 10 == 0:
        holder.models.gen.zero_grad()

        z = torch.randn(conf.loader.batch_size, 1, 96).to(device)
        tree = [z]

        fake_point = holder.models.gen(tree)
        G_fake = holder.models.disc(fake_point)
        G_fakem = G_fake.mean()

        g_loss = -G_fakem
        g_loss.backward()
        holder.optims.gen.step()


def training_procedure() -> None:
    holder: Holder = Holder()
    logger.warning(
        f"Starting training with state {str(OmegaConf.to_yaml(holder.state))}"
    )
    loader: QueuedDataLoader = QueuedDataLoader()
    train_log = TrainLog(holder)

    # Queue that batches
    loader.queue_epoch(n_skip_events=holder.state.processed_events)
    if not conf.debug and train_log.experiment.ended:
        logger.warning("Training has been completed, stopping.")
        loader.qfseq.stop()
        exit(0)
    try:
        while not early_stopping(holder.state):
            # switch model in training mode
            holder.models.train()
            for _ in tqdm(
                range(conf.training.validation_interval), postfix="training"
            ):
                holder.state.batch_start_time = time.time()
                try:
                    batch = next(loader.qfseq)
                except StopIteration:
                    # If there is no next batch go to the next epoch
                    if not conf.debug:
                        train_log.experiment.log_epoch_end(
                            holder.state["epoch"],
                            step=holder.state["grad_step"],
                        )
                    logger.warning("New epoch!")
                    holder.state.epoch += 1
                    holder.state.ibatch = 0
                    loader.queue_epoch(n_skip_events=holder.state.processed_events)
                    batch = next(loader.qfseq)
                batch = move_batch_to_device(batch, device)
                holder.state.time_io_done = time.time()
                training_step(batch, holder)
                holder.state.time_training_done = time.time()
                train_log.write_trainstep_logs()
                holder.state.ibatch += 1
                holder.state.processed_events += conf.loader.batch_size
                holder.state["grad_step"] += 1

            validate(holder, loader, train_log)
            holder.save_checkpoint()
        # Stopping
        holder.save_checkpoint()
        train_log.writer.flush()
        train_log.writer.close()
        logger.warning("Early Stopping criteria fulfilled")
        OmegaConf.save(holder.state, conf.path.complete_state)
        if not conf.debug:
            train_log.experiment.log_other("ended", True)
            train_log.experiment.end()
    except Exception as error:
        logger.error("Error detected, stopping qfseq.")
        loader.qfseq.stop()
        raise error
    loader.qfseq.stop()
    train_log.experiment.end()
    exit(0)
