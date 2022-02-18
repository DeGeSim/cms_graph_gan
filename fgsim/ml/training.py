"""Contain the training procedure, access point from __main__"""

import signal
import time
import traceback

from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataLoader
from fgsim.io.sel_seq import Batch
from fgsim.ml.early_stopping import early_stopping
from fgsim.ml.holder import Holder
from fgsim.ml.validation import validate
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.utils.memory import gpu_mem_monitor


def training_step(
    batch: Batch,
    holder: Holder,
) -> None:
    # set all optimizers to a 0 gradient
    holder.optims.zero_grad()
    # generate a new batch with the generator
    with gpu_mem_monitor("disc_training_points"):
        holder.reset_gen_points()
    with gpu_mem_monitor("disc_training_forward"):
        holder.losses.disc(holder, batch)
    with gpu_mem_monitor("disc_training_grad"):
        holder.optims.disc.step()
    with gpu_mem_monitor("disc_training_reset"):
        holder.optims.disc.zero_grad()

    # generator
    if holder.state.grad_step % conf.training.disc_steps_per_gen_step == 0:
        # generate a new batch with the generator, but with
        # points thought the generator this time
        with gpu_mem_monitor("gen_training_points"):
            holder.reset_gen_points_w_grad()
        with gpu_mem_monitor("gen_training_forward"):
            holder.losses.gen(holder, batch)
        with gpu_mem_monitor("gen_training_grad"):
            holder.optims.gen.step()
        with gpu_mem_monitor("gen_trainingreset"):
            holder.models.gen.zero_grad()


def training_procedure() -> None:
    holder: Holder = Holder()
    train_log: TrainLog = holder.train_log
    loader: QueuedDataLoader = QueuedDataLoader()

    sigterm_handler = SigTermHandel(holder, loader)
    # Queue that batches
    loader.queue_epoch(n_skip_events=holder.state.processed_events)

    exitcode = 0
    try:
        while not early_stopping(holder.state):
            if holder.state.grad_step == 0:
                holder.models.eval()
                validate(holder, loader)
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
                with gpu_mem_monitor("batch"):
                    batch = batch.to(device)
                holder.state.time_io_done = time.time()
                training_step(batch, holder)
                holder.state.time_training_done = time.time()
                train_log.write_trainstep_logs()
                holder.state.processed_events += conf.loader.batch_size
                holder.state["grad_step"] += 1
                holder.checkpoint_after_time()
            holder.models.eval()
            validate(holder, loader)
            holder.save_checkpoint()
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
        del sigterm_handler
        loader.qfseq.stop()
        exit(exitcode)


class SigTermHandel:
    def __init__(self, holder: Holder, loader: QueuedDataLoader) -> None:
        self.holder = holder
        self.loader = loader
        signal.signal(signal.SIGTERM, self.handle)
        signal.signal(signal.SIGINT, self.handle)
        print("Handle Initialized")

    def handle(self, _signo, _stack_frame):
        print("SIGTERM detected")
        self.holder.save_checkpoint()
        self.loader.qfseq.stop()
        exit()
