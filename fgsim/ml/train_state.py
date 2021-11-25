from comet_ml.experiment import BaseExperiment
from torch.utils.tensorboard import SummaryWriter

from fgsim.config import conf
from fgsim.ml.holder import Holder
from fgsim.monitor import setup_experiment, setup_writer


class TrainLog:
    """Initialized with the `holder`, provides the logging with cometml/tensorboard."""

    def __init__(self, holder: Holder):
        self.holder = holder
        self.writer: SummaryWriter = setup_writer()
        self.experiment: BaseExperiment = setup_experiment(holder)

    def write_trainstep_logs(self) -> None:
        if conf.debug:
            return
        traintime = (
            self.holder.state.time_training_done
            - self.holder.state.batch_start_time
        )
        iotime = self.holder.state.time_io_done - self.holder.state.batch_start_time
        utilisation = 1 - iotime / traintime

        # Tensorboard
        self.writer.add_scalar(
            "batchtime",
            traintime,
            self.holder.state["grad_step"],
        )
        self.writer.add_scalar(
            "utilisation",
            utilisation,
            self.holder.state["grad_step"],
        )
        self.writer.add_scalar(
            "processed_events",
            self.holder.state.processed_events,
            self.holder.state["grad_step"],
        )
        self.writer.add_scalar(
            "loss", self.holder.state.loss, self.holder.state["grad_step"]
        )
        self.writer.flush()

        # Comet.ml
        self.experiment.log_metric(
            "loss",
            self.holder.state.loss,
            step=self.holder.state["grad_step"],
            epoch=self.holder.state["epoch"],
        )
        self.experiment.log_metric(
            "utilisation",
            utilisation,
            step=self.holder.state["grad_step"],
            epoch=self.holder.state["epoch"],
        )
        self.experiment.log_metric(
            "batchtime",
            traintime,
            step=self.holder.state["grad_step"],
            epoch=self.holder.state["epoch"],
        )
        self.experiment.log_metric(
            "processed_events",
            self.holder.state.processed_events,
            step=self.holder.state["grad_step"],
            epoch=self.holder.state["epoch"],
        )

        #  self.experiment.log_histogram(
        #      experiment, gradmap, epoch * steps_per_epoch, prefix="gradient"
        #  )
