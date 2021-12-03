"""Contain the training procedure, access point from __main__"""

import time

from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import BatchType, QueuedDataLoader
from fgsim.ml.early_stopping import early_stopping
from fgsim.ml.holder import Holder, start_training_state
from fgsim.ml.validate import validate
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.utils.batch_utils import move_batch_to_device


def training_step(
    batch: BatchType,
    holder: Holder,
) -> None:
    # set all optimizers to a 0 gradient
    holder.optims.zero_grad()
    # generate a new batch with the generator
    holder.reset_gen_points()
    d_loss = holder.losses.disc(holder, batch)
    d_loss.backward()
    holder.optims.disc.step()

    # generator
    if holder.state.ibatch % conf.training.disc_steps_per_gen_step == 0:
        holder.models.gen.zero_grad()
        # generate a new batch with the generator, but with
        # points thought the generator this time
        holder.reset_gen_points_w_grad()
        g_loss = holder.losses.gen(holder, batch)
        g_loss.backward()
        holder.optims.gen.step()


def training_procedure() -> None:
    state = start_training_state()
    train_log = TrainLog(state)
    holder: Holder = Holder(state, train_log)
    loader: QueuedDataLoader = QueuedDataLoader()

    # Queue that batches
    loader.queue_epoch(n_skip_events=holder.state.processed_events)

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
                    train_log.next_epoch()
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

            validate(holder, loader)
            holder.save_checkpoint()
        # Stopping
        holder.state.complete = True
        train_log.end()
        holder.save_checkpoint()
        exit(0)

    except Exception as error:
        logger.error("Error detected, stopping qfseq.")
        loader.qfseq.stop()
        raise error
    loader.qfseq.stop()
    train_log.experiment.end()
    exit(0)
