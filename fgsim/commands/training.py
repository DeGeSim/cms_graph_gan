"""Contain the training procedure, access point from __main__"""

import time
import traceback

import torch
from pytorch_lightning.lite import LightningLite
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.pldata import PLDataFromQDL
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.io.sel_seq import Batch
from fgsim.ml.early_stopping import early_stopping
from fgsim.ml.holder import Holder
from fgsim.ml.validation import validate
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog


def training_step(
    batch: Batch,
    holder: Holder,
) -> None:
    # set all optimizers to a 0 gradient
    holder.optims.zero_grad()
    # generate a new batch with the generator
    holder.reset_gen_points()
    holder.losses.disc(holder, batch)
    holder.optims.disc.step()
    holder.optims.disc.zero_grad()

    # generator
    if holder.state.grad_step % conf.training.disc_steps_per_gen_step == 0:
        # generate a new batch with the generator, but with
        # points thought the generator this time
        holder.reset_gen_points_w_grad()
        holder.losses.gen(holder, batch)
        holder.optims.gen.step()
        holder.models.gen.zero_grad()


# class EpochEndCallBack(pl.callbacks.Callback):
#     def __init__(self) -> None:
#         super().__init__()


class Lite(LightningLite):
    def run(self):
        holder: Holder = Holder()
        if early_stopping(holder.history):
            exit()
        # train_log: TrainLog = holder.train_log

        optimizer = torch.optim.Adam(holder.models.parameters())
        model, optimizer = self.setup(holder.models, optimizer)

        loader: QueuedDataLoader = QueuedDataLoader()
        pl_data = PLDataFromQDL(loader)

        dataloader = self.setup_dataloaders(
            pl_data.train_dataloader()
        )  # Scale your dataloaders

        model.train()
        while True:
            loader.queue_epoch(n_skip_events=holder.state.processed_events)
            for batch in dataloader:
                optimizer.zero_grad()
                loss = model(batch)
                self.backward(loss)  # instead of loss.backward()
                optimizer.step()
                holder.state.processed_events += conf.loader.batch_size


def training_procedure() -> None:
    # Lite().run()
    # foo = PLDataFromQDL(loader)
    holder: Holder = Holder()
    if early_stopping(holder.history):
        exit()
    train_log: TrainLog = holder.train_log
    loader: QueuedDataLoader = QueuedDataLoader()

    # Queue that batches
    loader.queue_epoch(n_skip_events=holder.state.processed_events)

    exitcode = 0
    try:
        if not holder.checkpoint_loaded and not conf.debug:
            holder.models.eval()
            validate(holder, loader)
            holder.save_checkpoint()
        while not early_stopping(holder.history):
            # switch model in training mode
            holder.models.train()
            for _ in tqdm(
                range(conf.validation.interval),
                postfix=f"training from {holder.state.grad_step}",
            ):
                holder.state.batch_start_time = time.time()
                try:
                    batch = next(loader.qfseq)
                except StopIteration:
                    # If there is no next batch go to the next epoch
                    train_log.next_epoch()
                    holder.state.epoch += 1
                    loader.queue_epoch(n_skip_events=holder.state.processed_events)
                    batch = next(loader.qfseq)
                batch = batch.to(device)
                holder.state.time_io_done = time.time()
                training_step(batch, holder)
                holder.state.time_training_done = time.time()
                train_log.write_trainstep_logs()
                holder.state.processed_events += conf.loader.batch_size
                holder.state.grad_step += 1
                holder.checkpoint_after_time()
            holder.models.eval()
            validate(holder, loader)
        # Stopping
        holder.state.complete = True
        train_log.end()
        holder.save_checkpoint()
    except Exception as error:
        logger.error("Error detected, stopping qfseq.")
        exitcode = 1
        logger.error(error)
        traceback.print_exc()
    finally:
        loader.qfseq.stop()
        exit(exitcode)
