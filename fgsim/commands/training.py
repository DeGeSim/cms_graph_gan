"""Contain the training procedure, access point from __main__"""

import time
import traceback

from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import loader_info
from fgsim.ml.early_stopping import early_stopping
from fgsim.ml.holder import Holder
from fgsim.ml.smoothing import smooth_features
from fgsim.ml.validation import validate
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog


class Trainer:
    def __init__(self, holder: Holder) -> None:
        self.holder = holder
        if early_stopping(self.holder.history):
            exit()
        self.train_log: TrainLog = self.holder.train_log
        self.loader: QueuedDataset = QueuedDataset(loader_info)

        if not self.holder.checkpoint_loaded and not conf.debug and not conf.ray:
            self.validation_step()
            self.holder.save_checkpoint()

    def training_loop(self):
        while (
            not early_stopping(self.holder.history)
            and self.holder.state.epoch < conf.training.max_epochs
        ):
            self.train_epoch()

        self.post_training()

    def train_epoch(self):
        self.pre_epoch()
        istep_start = (
            self.holder.state.processed_events
            // conf.loader.batch_size
            % self.loader.n_grad_steps_per_epoch
        )
        for istep, batch in tqdm(
            zip(
                range(istep_start, self.loader.n_grad_steps_per_epoch + 1),
                self.loader.qfseq,
            ),
            initial=istep_start,
            total=self.loader.n_grad_steps_per_epoch,
            mininterval=5.0,
            desc=f"Epoch {self.holder.state.epoch}",
        ):
            batch = self.pre_training_step(batch)
            self.training_step(batch)
            self.post_training_step()
        self.validation_step()
        self.post_epoch()

    def pre_training_step(self, batch):
        if conf.training.smooth_features:
            batch.x = smooth_features(batch.x)
        batch = batch.to(device)
        self.holder.state.time_io_end = time.time()
        return batch

    def training_step(self, batch) -> None:
        # generate a new batch with the generator
        self.holder.optims.disc.zero_grad(set_to_none=True)
        self.holder.reset_gen_points()
        self.holder.losses.disc(self.holder, batch)
        self.holder.optims.disc.step()

        # generator
        if self.holder.state.grad_step % conf.training.disc_steps_per_gen_step == 0:
            # generate a new batch with the generator, but with
            # points thought the generator this time
            self.holder.models.gen.zero_grad(set_to_none=True)
            self.holder.reset_gen_points_w_grad()
            self.holder.losses.gen(self.holder, batch)
            self.holder.optims.gen.step()

    def post_training_step(self):
        self.holder.state.processed_events += conf.loader.batch_size
        self.holder.state.time_train_step_end = time.time()
        if self.holder.state.grad_step % conf.training.log_interval == 0:
            # aggregate the losses that have accumulated since the last time
            # and logg them
            ldict = {
                lpart.name: lpart.metric_aggr.aggregate()
                for lpart in self.holder.losses
            }
            for pname, ploosd in ldict.items():
                for lname, lossval in ploosd.items():
                    self.train_log.log_metric(f"train.{pname}.{lname}", lossval)
            # Also log training speed
            self.train_log.write_trainstep_logs()
        if not conf.ray:
            self.holder.checkpoint_after_time()
        self.holder.state.grad_step += 1
        self.holder.state.time_train_step_start = time.time()

    def validation_step(self):
        self.holder.models.eval()
        validate(self.holder, self.loader)
        self.holder.models.train()
        self.holder.state.time_train_step_start = time.time()

    def pre_epoch(self):
        self.loader.queue_epoch(n_skip_events=self.holder.state.processed_events)

    def post_epoch(self):
        self.train_log.next_epoch()

    def post_training(self):
        self.holder.state.complete = True
        self.train_log.end()
        if not conf.ray:
            self.holder.save_checkpoint()


def training_procedure() -> None:
    trainer = Trainer(Holder())
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
