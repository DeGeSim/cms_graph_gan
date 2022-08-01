from typing import Dict

from comet_ml.experiment import BaseExperiment
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from fgsim.config import conf
from fgsim.monitoring.logger import logger
from fgsim.monitoring.monitor import get_experiment, get_writer


class TrainLog:
    """Initialized with the `holder`, provides the logging with cometml/tensorboard."""

    def __init__(self, state, history):
        self.state: DictConfig = state
        self.history: Dict = history
        if conf.debug and conf.command != "test":
            return
        self.writer: SummaryWriter = get_writer()
        self.experiment: BaseExperiment = get_experiment(self.state)

    def log_model_graph(self, model):
        if conf.debug:
            return
        self.experiment.set_model_graph(str(model))

    def write_trainstep_logs(self) -> None:
        if not all(
            [
                hasattr(self.state, time)
                for time in [
                    "time_train_step_end",
                    "time_train_step_start",
                    "time_io_end",
                ]
            ]
        ):
            return
        traintime = (
            self.state.time_train_step_end - self.state.time_train_step_start
        )
        iotime = self.state.time_io_end - self.state.time_train_step_start
        utilisation = 1 - iotime / traintime

        self.log_loss("other.batchtime", traintime)
        self.log_loss("other.utilisation", utilisation)
        self.log_loss("other.processed_events", self.state.processed_events)

    def log_loss(self, lossname: str, loss) -> None:
        if conf.debug:
            return
        loss = float(loss)
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
        self.state["epoch"] += 1

    def end(self) -> None:
        if conf.debug:
            return
        self.writer.flush()
        self.writer.close()
        logger.warning("Early Stopping criteria fulfilled")
        if not conf.debug:
            self.experiment.log_other("ended", True)
            self.experiment.end()
