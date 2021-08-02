from dataclasses import dataclass
from typing import Optional

import omegaconf
from comet_ml.experiment import BaseExperiment
from torch.utils.tensorboard import SummaryWriter

from ..config import conf
from ..io.queued_dataset import QueuedDataLoader
from .holder import ModelHolder


@dataclass
class TrainState:
    """This object holds everything needed in the training and is used
    to access all needes objects. The state member is holder.state."""

    holder: ModelHolder
    state: omegaconf.omegaconf.Type
    loader: Optional[QueuedDataLoader]
    writer: Optional[SummaryWriter]
    experiment: Optional[BaseExperiment]

    def write_trainstep_logs(self) -> None:
        if conf.debug:
            return
        traintime = self.state.time_training_done - self.state.batch_start_time
        iotime = self.state.time_io_done - self.state.batch_start_time
        utilisation = 1 - iotime / traintime

        # Tensorboard
        self.writer.add_scalar(
            "batchtime",
            traintime,
            self.state["grad_step"],
        )
        self.writer.add_scalar(
            "utilisation",
            utilisation,
            self.state["grad_step"],
        )
        self.writer.add_scalar(
            "processed_events",
            self.state.processed_events,
            self.state["grad_step"],
        )
        self.writer.add_scalar("loss", self.state.loss, self.state["grad_step"])
        self.writer.flush()

        # Comet.ml
        self.experiment.log_metric(
            "loss",
            self.state.loss,
            step=self.state["grad_step"],
            epoch=self.state["epoch"],
        )
        self.experiment.log_metric(
            "utilisation",
            utilisation,
            step=self.state["grad_step"],
            epoch=self.state["epoch"],
        )
        self.experiment.log_metric(
            "batchtime",
            traintime,
            step=self.state["grad_step"],
            epoch=self.state["epoch"],
        )
        self.experiment.log_metric(
            "processed_events",
            self.state.processed_events,
            step=self.state["grad_step"],
            epoch=self.state["epoch"],
        )
