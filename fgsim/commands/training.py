"""Contain the training procedure, access point from __main__"""

import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.io.lightning_dataset import PLDataFromQDS
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import loader_info
from fgsim.ml.early_stopping import early_stopping
from fgsim.ml.holder import Holder
from fgsim.ml.smoothing import smooth_features
from fgsim.monitoring.train_log import TrainLog


class LitModel(pl.LightningModule):
    def __init__(self, loader: QueuedDataset) -> None:
        super().__init__()

        self.holder: Holder = Holder()
        if early_stopping(self.holder.history):
            exit()
        self.train_log: TrainLog = self.holder.train_log
        self.loader = loader

        if not self.holder.checkpoint_loaded and not conf.debug:
            self.holder.save_checkpoint()

    def configure_optimizers(self):
        return [self.holder.optims.gen, self.holder.optims.disc], []

    def training_step(self, batch, batch_idx, optimizer_idx) -> torch.Tensor:

        # set all optimizers to a 0 gradient
        self.holder.optims.zero_grad()
        if optimizer_idx == 1:
            # generate a new batch with the generator
            self.holder.reset_gen_points()
            return self.holder.losses.disc(self.holder, batch)

        # generator
        elif optimizer_idx == 0:
            # generate a new batch with the generator, but with
            # points thought the generator this time
            self.holder.reset_gen_points_w_grad()
            return self.holder.losses.gen(self.holder, batch)
        else:
            raise Exception

    def validation_step(self, sim_batch, batch_idx):
        gen_graphs = []
        for _ in range(conf.loader.validation_set_size // conf.loader.batch_size):
            self.holder.reset_gen_points()
            for igraph in range(conf.loader.batch_size):
                gen_graphs.append(self.holder.gen_points.get_example(igraph))
        gen_batch = Batch.from_data_list(gen_graphs)

        D_sim = self.holder.models.disc(sim_batch)
        D_gen = self.holder.models.disc(gen_batch)
        self.holder.val_loss(gen_batch, sim_batch, D_sim, D_gen)
        self.holder.val_loss.log_metrics()

        min_stop_crit = min(self.holder.history["stop_crit"])
        if min_stop_crit == self.holder.history["stop_crit"][-1]:
            self.holder.state.best_grad_step = self.holder.state["grad_step"]
            self.holder.best_model_state = self.holder.models.state_dict()

            if not conf.debug:
                self.holder.train_log.experiment.log_metric(
                    "min_stop_crit", min_stop_crit
                )
                self.holder.train_log.experiment.log_metric(
                    "best_grad_step", self.holder.state["grad_step"]
                )
                self.holder.train_log.experiment.log_metric(
                    "best_epoch", self.holder.state["epoch"]
                )
        if not conf.debug:
            self.holder.train_log.writer.flush()


def training_procedure() -> None:
    qds = QueuedDataset(loader_info)
    model = LitModel(qds)

    pldata = PLDataFromQDS(qds)
    trainer = pl.Trainer(callbacks=[AllCallbacks()], max_epochs=-1)
    trainer.fit(
        model=model,
        train_dataloaders=pldata.train_dataloader(),
        val_dataloaders=pldata.val_dataloader(),
    )


class AllCallbacks(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.holder.state.time_io_done = time.time()
        pl_module.holder.state.batch_start_time = time.time()
        if conf.training.smooth_features:
            batch.x = smooth_features(batch.x)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.holder.state.processed_events += conf.loader.batch_size
        pl_module.holder.state.time_training_done = time.time()
        if pl_module.holder.state.grad_step % conf.training.log_interval == 0:
            # aggregate the losses that have accumulated since the last time
            # and logg them
            ldict = {
                lpart.name: lpart.metric_aggr.aggregate()
                for lpart in pl_module.holder.losses
            }
            for pname, ploosd in ldict.items():
                for lname, lossval in ploosd.items():
                    pl_module.train_log.log_loss(f"train.{pname}.{lname}", lossval)
            # Also log training speed
            pl_module.train_log.write_trainstep_logs()
        pl_module.holder.checkpoint_after_time()
        pl_module.holder.state.grad_step += 1

    def on_train_epoch_start(self, trainer, pl_module):
        # Queue that batches
        pl_module.loader.queue_epoch(
            n_skip_events=pl_module.holder.state.processed_events
        )

    def on_train_epoch_end(self, trainer, pl_module):
        # If there is no next batch go to the next epoch
        pl_module.train_log.next_epoch()
        pl_module.loader.queue_epoch(
            n_skip_events=pl_module.holder.state.processed_events
        )

    def on_exception(self, trainer, pl_module, exception):
        pl_module.loader.qfseq.stop()

    def on_fit_end(self, trainer, pl_module):
        pl_module.holder.state.complete = True
        pl_module.train_log.end()
        pl_module.holder.save_checkpoint()


# def training_loop(self):
#     while not early_stopping(self.holder.history):
#         # switch model in training mode
#         self.holder.models.train()
#         for _ in tqdm(
#             range(conf.validation.interval),
#             postfix=f"train {self.holder.state.grad_step}",
#         ):
#
#             try:
#                 batch = next(self.loader.qfseq)
#             except StopIteration:
#                 self.post_epoch()
#                 batch = next(self.loader.qfseq)
#             batch = self.pre_training_step(batch)
#             self.training_step(batch)
#             self.post_training_step()
#         self.validation_step()
#     # Stopping
#     self.post_training()
