"""This modules manages all objects that need to be available for the training:
Subnetworks, losses and optimizers. The Subnetworks and losses are dynamically imported,
depending on the config. Contains the code for checkpointing of model and optimzer status."""


import os

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from fgsim.config import conf, device
from fgsim.ml.loss import LossesCol
from fgsim.ml.network import SubNetworkCollector
from fgsim.ml.optim import OptimCol
from fgsim.ml.val_metrics import ValidationMetrics
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.types import Batch
from fgsim.utils.check_for_nans import contains_nans
from fgsim.utils.memory import gpu_mem_monitor
from fgsim.utils.push_to_old import push_to_old


def start_training_state() -> DictConfig:
    return OmegaConf.create(
        {
            "epoch": 0,
            "processed_events": 0,
            "ibatch": 0,
            "grad_step": 0,
            "losses": {
                snwname: {lossname: [] for lossname in snwconf.losses}
                for snwname, snwconf in conf.models.items()
            },
            "val_metrics": {},
            "complete": False,
        }
    )


class Holder:
    """ "This class holds the models, the loss functions and the optimizers.
    It manages the checkpointing and holds a member 'state' that contains
    information about the current state of the training"""

    # Nameing convention snw = snw
    def __init__(self) -> None:
        self.state: DictConfig = start_training_state()
        self.train_log = TrainLog(self.state)

        self.models: SubNetworkCollector = SubNetworkCollector(conf.models)
        self.train_log.log_model_graph(self.models)
        self.losses: LossesCol = LossesCol(self.train_log)
        self.val_loss: ValidationMetrics = ValidationMetrics(self.train_log)
        self.optims: OptimCol = OptimCol(conf.models, self.models.get_par_dict())

        # try to load a check point
        self.checkpoint_loaded = False
        self.__load_checkpoint()

        with gpu_mem_monitor("models"):
            self.models = self.models.float().to(device)
        # Hack to move the optim parameters to the correct device
        # https://github.com/pytorch/pytorch/issues/8741
        with gpu_mem_monitor("optims"):
            self.optims.load_state_dict(self.optims.state_dict())

        state_filtered = OmegaConf.create(
            {key: self.state[key] for key in self.state if "loss" not in key}
        )
        logger.warning(
            f"Starting training with state {str(OmegaConf.to_yaml(state_filtered))}"
        )

        if self.state.complete and conf["command"] == "train":
            logger.warning("Training has been completed, stopping.")
            exit(0)

        # Keep the generated samples ready, to be accessed by the losses
        self.gen_points: Batch = None
        self.gen_points_w_grad: Batch = None

    def __load_checkpoint(self):
        if not (
            os.path.isfile(conf.path.state) and os.path.isfile(conf.path.checkpoint)
        ):
            logger.warning("Proceeding without loading checkpoint.")
            return

        self.state = OmegaConf.load(conf.path.state)
        # Once the state has been loaded from the checkpoint,
        #  update the logger state
        self.train_log.state = self.state
        checkpoint = torch.load(conf.path.checkpoint, map_location=device)

        assert not contains_nans(checkpoint["models"])[0]
        assert not contains_nans(checkpoint["best_model"])[0]

        self.models.load_state_dict(checkpoint["models"])
        self.optims.load_state_dict(checkpoint["optims"])
        self.best_model_state = checkpoint["best_model"]

        logger.warning(
            "Loading model from checkpoint at"
            + f" epoch {self.state['epoch']}"
            + f" batch {self.state['ibatch']}"
            + f" grad_step {self.state['grad_step']}."
        )

    def select_best_model(self):
        self.models.load_state_dict(self.best_model_state)
        self.models = self.models.float().to(device)

    def save_checkpoint(self):
        push_to_old(conf.path.checkpoint, conf.path.checkpoint_old)
        torch.save(
            {
                "models": self.models.state_dict(),
                "optims": self.optims.state_dict(),
                "best_model": self.best_model_state,
            },
            conf.path.checkpoint,
        )
        push_to_old(conf.path.state, conf.path.state_old)
        OmegaConf.save(config=self.state, f=conf.path.state)

    # Define the methods, that equip the with the generated batches
    def gen_noise(self, requires_grad=False) -> torch.Tensor:
        n_features = (
            conf.loader.n_features + conf.models.gen.params.n_hidden_features
        )
        return torch.randn(
            conf.loader.batch_size, 1, n_features, requires_grad=requires_grad
        ).to(device)

    def reset_gen_points(self) -> None:
        with torch.no_grad():
            z = self.gen_noise()
            self.gen_points = self.models.gen(z)

    def reset_gen_points_w_grad(self) -> None:
        z = self.gen_noise(True)
        self.gen_points_w_grad = self.models.gen(z)
