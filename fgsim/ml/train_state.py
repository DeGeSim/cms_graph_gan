from dataclasses import dataclass

import omegaconf
from comet_ml.experiment import BaseExperiment
from torch.utils.tensorboard import SummaryWriter

from ..io.queued_dataset import QueuedDataLoader
from .holder import ModelHolder


@dataclass
class TrainState:
    """This object holds everything needed in the training and is used
    to access all needes objects. The state member is holder.state."""

    holder: ModelHolder
    state: omegaconf.omegaconf.Type
    loader: QueuedDataLoader
    writer: SummaryWriter
    experiment: BaseExperiment

    def writelogs(self) -> None:
        times = [
            self.state.batch_start_time,
            self.state.time_io_done,
            self.state.time_training_done,
            self.state.time_validation_done,
            self.state.time_early_stopping_done,
        ]
        (iotime, traintime, valtime, esttime) = [
            times[itime] - times[itime - 1] for itime in range(1, len(times))
        ]
        timesD = {
            "iotime": iotime,
            "traintime": traintime,
            "valtime": valtime,
            "esttime": esttime,
        }

        self.writer.add_scalars(
            "times",
            timesD,
            self.state["grad_step"],
        )
        self.writer.add_scalar("loss", self.state.loss, self.state["grad_step"])
        self.writer.flush()

        self.experiment.log_metrics(
            dic=timesD,
            prefix="times",
            step=self.state.grad_step,
            epoch=self.state.epoch,
        )
        self.experiment.log_metric("loss", self.state.loss, self.state["grad_step"])
