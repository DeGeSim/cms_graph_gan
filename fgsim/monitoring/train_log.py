from typing import Dict, Union

import torch
import wandb
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from fgsim.config import conf
from fgsim.monitoring.monitor import exp_orga_wandb, get_experiment

if not conf.ray:
    from comet_ml.experiment import BaseExperiment


class TrainLog:
    """Initialized with the `holder`, provides the logging with cometml/tensorboard.
    """

    def __init__(self, state, history):
        self.state: DictConfig = state
        self.history: Dict = history
        self.use_tb = not conf.debug or conf.command == "test"
        self.use_comet = False
        # (not conf.debug or conf.command == "test") and not conf.ray
        self.use_wandb = (not conf.debug or conf.command == "test") and not conf.ray
        if self.use_tb:
            self.writer: SummaryWriter = SummaryWriter(conf.path.tensorboard)

        if self.use_comet:
            self.experiment: BaseExperiment = get_experiment(self.state)

        if self.use_tb:
            self.writer.add_scalar(
                "epoch",
                self.state["epoch"],
                self.state["grad_step"],
                new_style=True,
            )

        if self.use_wandb:
            self.wandb_run = wandb.init(
                id=exp_orga_wandb[conf["hash"]],
                resume="must",
                dir=conf.path.run_path,
                project=conf.comet_project_name,
                job_type=conf.command,
            )
            wandb.log(
                data={"other/epoch": self.state["epoch"]},
                step=self.state["grad_step"],
            )

    def log_model_graph(self, model):
        if self.use_comet:
            self.experiment.set_model_graph(str(model))
        if self.use_wandb:
            wandb.watch(model)

    def log_metrics(
        self,
        metrics_dict: dict[str, Union[float, torch.Tensor]],
        step=None,
        epoch=None,
        prefix=None,
    ):
        if conf.debug and conf.command != "test":
            return
        if step is None:
            step = self.state["grad_step"]
            epoch = self.state["epoch"]
        if epoch is None:
            raise Exception

        if prefix is not None:
            metrics_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}

        self._log_metrics_comet(metrics_dict, step, epoch)
        self._log_metrics_tb(metrics_dict, step, epoch)
        self._log_metrics_wandb(metrics_dict, step, epoch)

    def _log_metrics_comet(self, md: dict, step: int, epoch: int):
        if self.use_comet:
            self.experiment.log_metrics(
                md,
                step=step,
                epoch=epoch,
            )

    def _log_metrics_tb(self, md: dict, step: int, epoch: int):
        if self.use_tb:
            for name, value in md.items():
                self.writer.add_scalar(name, value, step, new_style=True)

    def _log_metrics_wandb(self, md: dict, step: int, epoch: int):
        if self.use_wandb:
            wandb.log(md | {"epoch": epoch}, step=step)

    def log_figure(
        self,
        figure_name,
        figure,
        overwrite=False,
        step=None,
    ):
        if step is None:
            step = self.state["grad_step"]
        if self.use_tb:
            self.writer.add_figure(tag=figure_name, figure=figure, global_step=step)
        if self.use_comet:
            self.experiment.log_figure(
                figure_name=figure_name,
                figure=figure,
                overwrite=overwrite,
                step=step,
            )
        if self.use_wandb:
            wandb.log(data={figure_name: wandb.Image(figure)}, step=step)

    def log_test_metrics(
        self,
        metrics_dict: dict[str, Union[float, torch.Tensor]],
        step: int,
        epoch: int,
        prefix: str,
    ):
        metrics_dict = {f"{prefix}/{k}": v for k, v in metrics_dict.items()}
        self._log_metrics_comet(metrics_dict, step, epoch)
        self._log_metrics_tb(metrics_dict, step, epoch)
        self._log_metrics_wandb(metrics_dict, step, epoch)
        if self.use_wandb:
            for k, v in metrics_dict.items():
                self.wandb_run.summary[k] = v

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

        self.log_metrics(
            {
                "batchtime": traintime,
                "utilisation": utilisation,
                "processed_events": self.state.processed_events,
            },
            prefix="speed",
        )

    def next_epoch(self) -> None:
        self.state["epoch"] += 1
        if self.use_comet:
            self.experiment.log_epoch_end(
                self.state["epoch"],
                step=self.state["grad_step"],
            )

        if self.use_tb:
            self.writer.add_scalar(
                "epoch",
                self.state["epoch"],
                self.state["grad_step"],
                new_style=True,
            )
        if self.use_wandb:
            wandb.log(
                data={"epoch": self.state["epoch"]},
                step=self.state["grad_step"],
            )

    def __del__(self):
        if self.use_tb:
            self.writer.flush()
            self.writer.close()

    def end(self) -> None:
        if self.use_comet:
            self.experiment.log_other("ended", True)
            self.experiment.end()
