"""Contain the training procedure, access point from __main__"""

import time

from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import Batch, QueuedDataLoader
from fgsim.ml.early_stopping import early_stopping
from fgsim.ml.holder import Holder
from fgsim.ml.validate import validate
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.utils.memory import gpu_mem_monitor

exitcode = 0


def training_step(
    batch: Batch,
    holder: Holder,
) -> None:
    # set all optimizers to a 0 gradient
    holder.optims.zero_grad()
    # generate a new batch with the generator
    with gpu_mem_monitor("gen_points", False):
        holder.reset_gen_points()

    holder.losses.disc(holder, batch)
    with gpu_mem_monitor("disc_grad"):
        holder.optims.disc.step()

    # generator
    if holder.state.ibatch % conf.training.disc_steps_per_gen_step == 0:
        holder.models.gen.zero_grad()
        # generate a new batch with the generator, but with
        # points thought the generator this time
        with gpu_mem_monitor("gen_points_w_grad", False):
            holder.reset_gen_points_w_grad()
        holder.losses.gen(holder, batch)
        with gpu_mem_monitor("gen_grad"):
            holder.optims.gen.step()


def training_procedure() -> None:
    holder: Holder = Holder()
    train_log: TrainLog = holder.train_log

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
                with gpu_mem_monitor("batch"):
                    batch = batch.to(device)
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
    except Exception as error:
        logger.error("Error detected, stopping qfseq.")
        exitcode = 1
        raise error
    finally:
        loader.qfseq.stop()
        exit(exitcode)
