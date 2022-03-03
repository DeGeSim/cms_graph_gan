from comet_ml.experiment import BaseExperiment
from torch.utils.tensorboard import SummaryWriter

from fgsim.config import conf
from fgsim.monitoring.logger import logger
from fgsim.monitoring.monitor import get_writer, setup_experiment


class TrainLog:
    """Initialized with the `holder`, provides the logging with cometml/tensorboard."""

    def __init__(self, state):
        if conf.debug:
            return
        self.state = state
        self.writer: SummaryWriter = get_writer()
        self.experiment: BaseExperiment = setup_experiment(self.state)

    def log_model_graph(self, model):
        if conf.debug:
            return
        self.experiment.set_model_graph(str(model))

    def write_trainstep_logs(self) -> None:
        if conf.debug:
            return
        traintime = self.state.time_training_done - self.state.batch_start_time
        iotime = self.state.time_io_done - self.state.batch_start_time
        utilisation = 1 - iotime / traintime

        self.writer.add_scalar(
            "batchtime",
            traintime,
            self.state["grad_step"],
        )
        self.experiment.log_metric(
            "batchtime",
            traintime,
            step=self.state["grad_step"],
            epoch=self.state["epoch"],
        )
        self.experiment.log_metric(
            "utilisation",
            utilisation,
            step=self.state["grad_step"],
            epoch=self.state["epoch"],
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
        self.experiment.log_metric(
            "processed_events",
            self.state.processed_events,
            step=self.state["grad_step"],
            epoch=self.state["epoch"],
        )

        #  self.experiment.log_histogram(
        #      experiment, gradmap, epoch * steps_per_epoch, prefix="gradient"
        #  )
        self.writer.flush()

    def log_loss(self, lossname: str, loss: float) -> None:
        if conf.debug:
            return
        self.writer.add_scalar(lossname, loss, self.state["grad_step"])
        self.experiment.log_metric(
            lossname,
            loss,
            step=self.state["grad_step"],
            epoch=self.state["epoch"],
        )

    def next_epoch(self) -> None:
        if conf.debug:
            return
        self.experiment.log_epoch_end(
            self.state["epoch"],
            step=self.state["grad_step"],
        )
        logger.warning("New epoch!")

    def end(self) -> None:
        if conf.debug:
            return
        self.writer.flush()
        self.writer.close()
        logger.warning("Early Stopping criteria fulfilled")
        if not conf.debug:
            self.experiment.log_other("ended", True)
            self.experiment.end()
