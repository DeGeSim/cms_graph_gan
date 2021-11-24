"""This modules manages all objects that need to be available for the training:
Subnetworks, losses and optimizers. The Subnetworks and losses are dynamically imported,
depending on the config. Contains the code for checkpointing of model and optimzer status."""


import os
from typing import Dict

from omegaconf import OmegaConf

from fgsim.config import conf
from fgsim.ml.network import SubNetwork, SubNetworkCollector
from fgsim.ml.optim import OptimCol
from fgsim.utils.logger import logger
from fgsim.utils.push_to_old import push_to_old


class ModelHolder:
    """ "This class holds the model the loss function and the optim.
    It manages the checkpointing and holds a member 'state' that contains
    information about the current state of the training"""

    def __init__(self) -> None:
        self.state = OmegaConf.create(
            {
                "epoch": 0,
                "processed_events": 0,
                "ibatch": 0,
                "grad_step": 0,
                "val_losses": [],
            }
        )
        self.submodels: Dict[str, SubNetwork] = {}
        self.model = SubNetworkCollector(self.submodels)
        for name, submodel in self.model.parts.items():
            self.submodels[name] = submodel

        self.optims = OptimCol(
            {name: submodel.optim for name, submodel in self.model.parts.items()}
        )

        self.__load_checkpoint()
        checkpoint_loaded = next(iter(self.submodels.values())).checkpoint_loaded
        assert all(
            checkpoint_loaded == e.checkpoint_loaded
            for e in self.submodels.values()
        ), (
            "Some parts of the model have been loaded from the checkpoint, others"
            " not "
        )
        if checkpoint_loaded:
            logger.warning(
                "Loading model from checkpoint at"
                + f" epoch {self.state['epoch']}"
                + f" batch {self.state['ibatch']}"
                + f" grad_step {self.state['grad_step']}."
            )
        else:
            logger.warning("Starting new training from scratch.")

    def __load_checkpoint(self):
        if not os.path.isfile(conf.path.state):
            logger.warning("Proceeding without loading checkpoint.")
            return

        self.state = OmegaConf.load(conf.path.state)

    def select_best_model(self):
        self.model.select_best_model()

    def save_checkpoint(self):
        push_to_old(conf.path.state, conf.path.state_old)
        OmegaConf.save(config=self.state, f=conf.path.state)
        self.model.save_checkpoint()


model_holder = ModelHolder()
