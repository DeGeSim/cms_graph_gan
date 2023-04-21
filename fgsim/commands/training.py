"""Contain the training procedure, access point from __main__"""

import signal
import sys
import time
import traceback

import torch
from queueflow import Sequence
from tqdm import tqdm

import wandb
from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import loader_info
from fgsim.ml.early_stopping import early_stopping
from fgsim.ml.holder import Holder
from fgsim.ml.smoothing import smooth_features
from fgsim.ml.validation import validate
from fgsim.monitoring import TrainLog, logger
from fgsim.utils.senderror import send_error


class Trainer:
    def __init__(self, holder: Holder) -> None:
        self.holder = holder
        self.train_log: TrainLog = self.holder.train_log
        self.loader: QueuedDataset = QueuedDataset(loader_info)
        self.val_interval = (
            conf.training.val.debug_interval
            if conf.debug
            else conf.training.val.interval
        )
        logger.info(f"Device: {torch.cuda.get_device_name()}")

    def training_loop(self):
        max_epochs = conf.training.max_epochs
        state = self.holder.state
        while state.epoch < max_epochs and not early_stopping(self.holder):
            self.train_epoch()
        else:
            if state.epoch >= max_epochs:
                logger.warning("Max Epochs surpassed")
            else:
                logger.warning("Early Stopping criteria fulfilled")
        self.post_training()

    def train_epoch(self):
        self.pre_epoch()
        tbar = tqdm(self.loader.qfseq, **self.tqdmkw())
        for batch in tbar:
            if self.holder.state.grad_step % self.val_interval == 0:
                self.validation_step()
                tbar.unpause()
            batch = self.pre_training_step(batch)
            self.training_step(batch)
            self.post_training_step()

        self.post_epoch()

    def pre_training_step(self, batch):
        if conf.training.smoothing.active:
            batch.x = smooth_features(batch.x, self.holder.state.grad_step)
        batch = batch.to(device)
        self.holder.state.time_io_end = time.time()
        return batch

    def training_step(self, batch) -> dict:
        # generate a new batch with the generator
        self.holder.optims.zero_grad(set_to_none=True)
        res = self.holder.pass_batch_through_model(batch, train_disc=True)
        self.holder.losses.disc(self.holder, **res)
        self.holder.optims.step("disc")

        # generator
        if self.holder.state.grad_step % conf.training.disc_steps_per_gen_step == 0:
            # generate a new batch with the generator, but with
            # points thought the generator this time
            self.holder.optims.zero_grad(set_to_none=True)
            res = self.holder.pass_batch_through_model(batch, train_gen=True)
            self.holder.losses.gen(self.holder, **res)
            self.holder.optims.step("gen")
        return res

    def post_training_step(self):
        self.holder.state.processed_events += conf.loader.batch_size
        self.holder.state.time_train_step_end = time.time()
        loss_hist = self.holder.history["losses"]

        if self.holder.state.grad_step % conf.training.log_interval == 0:
            # aggregate the losses that have accumulated since the last time
            ldict = {
                lpart.name: lpart.metric_aggr.aggregate()
                for lpart in self.holder.losses
            }
            # and log them & attacht them to the history (for early stopping)
            for pname, ploosd in ldict.items():
                self.train_log.log_metrics(
                    {
                        f"train/{pname}/{lname}": lossval
                        for lname, lossval in ploosd.items()
                    }
                )
                if pname not in loss_hist:
                    loss_hist[pname] = {}
                for lname, lossval in ploosd.items():
                    if lname not in loss_hist[pname]:
                        loss_hist[pname][lname] = []
                    loss_hist[pname][lname].append(lossval)

            # log the learning rates
            lr_dict = self.holder.optims.metric_aggr.aggregate()

            self.train_log.log_metrics(
                {f"train/{pname}/lr": plr for pname, plr in lr_dict.items()}
            )

            # Also log training speed
            self.train_log.write_trainstep_logs()
        if not conf.ray:
            self.holder.checkpoint_after_time()
        self.holder.state.grad_step += 1
        self.holder.state.time_train_step_start = time.time()

    def validation_step(self):
        if not conf.debug:
            logger.info("Validating")
            validate(self.holder, self.loader)
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

    def tqdmkw(self):
        kws = dict()
        kws["initial"] = (
            self.holder.state.processed_events
            // conf.loader.batch_size
            % self.loader.n_grad_steps_per_epoch
        )
        if conf.debug:
            kws["miniters"] = 5
            kws["mininterval"] = 1.0
        elif self.holder.state.epoch < 10:
            kws["miniters"] = 200
            kws["mininterval"] = 10.0
        else:
            kws["miniters"] = 1000
            kws["mininterval"] = 20.0
        kws["total"] = self.loader.n_grad_steps_per_epoch
        kws["desc"] = f"Epoch {self.holder.state.epoch}"
        return kws


def training_procedure() -> None:
    holder = Holder(device)
    trainer = Trainer(holder)
    term_handler = SigTermHander(holder, trainer.loader.qfseq)
    # Regular run
    if sys.gettrace() is not None:
        trainer.training_loop()
    # Debugger is running
    else:
        exitcode = 0
        try:
            trainer.training_loop()
        except Exception:
            exitcode = 1
            tb = traceback.format_exc()
            send_error(tb)
            logger.error(tb)
        finally:
            logger.error("Error detected, stopping qfseq.")
            if trainer.loader.qfseq.started:
                trainer.loader.qfseq.stop()
            exit(exitcode)
    del term_handler


class SigTermHander:
    def __init__(self, holder: Holder, qfseq: Sequence) -> None:
        self.qfseq = qfseq
        self.holder = holder
        signal.signal(signal.SIGTERM, self.handle)
        signal.signal(signal.SIGINT, self.handle)

    def handle(self, _signo, _stack_frame):
        print("SIGTERM detected, stopping qfseq")
        #  self.holder.save_checkpoint()
        self.qfseq.stop()
        self.holder.save_checkpoint()
        wandb.mark_preempting()

        exit()
