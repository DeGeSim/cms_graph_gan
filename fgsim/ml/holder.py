"""This modules manages all objects that need to be available for the training:
Subnetworks, losses and optimizers. The Subnetworks and losses are dynamically imported,
depending on the config. Contains the code for checkpointing of model and optimzer status."""


import os

import torch
from omegaconf import OmegaConf

from fgsim.config import conf, device
from fgsim.ml.loss import LossesCol
from fgsim.ml.network import SubNetworkCollector
from fgsim.ml.optim import OptimCol
from fgsim.utils.check_for_nans import contains_nans
from fgsim.utils.logger import logger
from fgsim.utils.push_to_old import push_to_old


class Holder:
    """ "This class holds the models, the loss functions and the optimizers.
    It manages the checkpointing and holds a member 'state' that contains
    information about the current state of the training"""

    def __init__(
        self,
    ) -> None:
        self.state: OmegaConf = OmegaConf.create(
            {
                "epoch": 0,
                "processed_events": 0,
                "ibatch": 0,
                "grad_step": 0,
                "val_losses": [],
            }
        )

        self.models: SubNetworkCollector = SubNetworkCollector(conf.models)
        self.losses: LossesCol = LossesCol(conf.models)
        self.optims: OptimCol = OptimCol(conf.models)

        # try to load a check point
        self.checkpoint_loaded = False
        self.__load_checkpoint()

        # Hack to move the optim parameters to the correct device
        self.models = self.models.float().to(device)
        # https://github.com/pytorch/pytorch/issues/8741
        self.optims.load_state_dict(self.optims.state_dict())

        self.__load_checkpoint()

    def __load_checkpoint(self):
        if not (
            os.path.isfile(conf.path.state) and os.path.isfile(conf.path.checkpoint)
        ):
            logger.warning("Proceeding without loading checkpoint.")
            return

        self.state = OmegaConf.load(conf.path.state)
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
