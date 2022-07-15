"""Contain the training procedure, access point from __main__"""

import time
import traceback

from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.ml.early_stopping import early_stopping
from fgsim.ml.holder import Holder
from fgsim.ml.smoothing import smooth_features
from fgsim.ml.validation import validate
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog


class Trainer:
    def __init__(self) -> None:
        self.holder: Holder = Holder()
        if early_stopping(self.holder.history):
            exit()
        self.train_log: TrainLog = self.holder.train_log
        self.loader: QueuedDataset = QueuedDataset()

        # Queue that batches
        self.loader.queue_epoch(n_skip_events=self.holder.state.processed_events)

        if not self.holder.checkpoint_loaded and not conf.debug:
            self.holder.models.eval()
            validate(self.holder, self.loader)
            self.holder.save_checkpoint()

    def training_loop(self):
        while not early_stopping(self.holder.history):
            # switch model in training mode
            self.holder.models.train()
            for _ in tqdm(
                range(conf.validation.interval),
                postfix=f"train {self.holder.state.grad_step}",
            ):
                self.holder.state.batch_start_time = time.time()
                try:
                    batch = next(self.loader.qfseq)
                except StopIteration:
                    self.post_epoch()
                    batch = next(self.loader.qfseq)
                batch = self.pre_training_step(batch)
                self.training_step(batch)
                self.post_training_step()
            self.validation_step()
        # Stopping
        self.post_training()

    def pre_training_step(self, batch):
        if conf.training.smooth_features:
            batch.x = smooth_features(batch.x)
        batch = batch.to(device)
        self.holder.state.time_io_done = time.time()
        return batch

    def training_step(self, batch) -> None:
        # set all optimizers to a 0 gradient
        self.holder.optims.zero_grad()
        # generate a new batch with the generator
        self.holder.reset_gen_points()
        self.holder.losses.disc(self.holder, batch)
        self.holder.optims.disc.step()
        self.holder.optims.disc.zero_grad()

        # generator
        if self.holder.state.grad_step % conf.training.disc_steps_per_gen_step == 0:
            # generate a new batch with the generator, but with
            # points thought the generator this time
            self.holder.reset_gen_points_w_grad()
            self.holder.losses.gen(self.holder, batch)
            self.holder.optims.gen.step()
            self.holder.models.gen.zero_grad()

    def post_training_step(self):
        self.holder.state.processed_events += conf.loader.batch_size
        self.holder.state.time_training_done = time.time()
        if self.holder.state.grad_step % conf.training.log_interval == 0:
            # aggregate the losses that have accumulated since the last time
            # and logg them
            ldict = {
                lpart.name: lpart.metric_aggr.aggregate()
                for lpart in self.holder.losses
            }
            for pname, ploosd in ldict.items():
                for lname, lossval in ploosd.items():
                    self.train_log.log_loss(f"train.{pname}.{lname}", lossval)
            # Also log training speed
            self.train_log.write_trainstep_logs()
        self.holder.checkpoint_after_time()
        self.holder.state.grad_step += 1

    def validation_step(self):
        validate(self.holder, self.loader)

    def post_epoch(self):
        # If there is no next batch go to the next epoch
        self.train_log.next_epoch()
        self.loader.queue_epoch(n_skip_events=self.holder.state.processed_events)

    def post_training(self):
        self.holder.state.complete = True
        self.train_log.end()
        self.holder.save_checkpoint()


def training_procedure() -> None:
    trainer = Trainer()
    exitcode = 0
    try:
        trainer.training_loop()
    except Exception as error:
        logger.error("Error detected, stopping qfseq.")
        exitcode = 1
        logger.error(error)
        traceback.print_exc()
    finally:
        trainer.loader.qfseq.stop()
        exit(exitcode)
