import importlib
import os

import torch
from omegaconf import OmegaConf

from ..config import conf, device
from ..utils.check_for_nans import contains_nans
from ..utils.count_parameters import count_parameters
from ..utils.logger import logger

# Import the specified model
ModelClass = importlib.import_module(
    f"..models.{conf.model.name}", "fgsim.models"
).ModelClass


class ModelHolder:
    """ "This class holds the model the loss function and the optimizer.
    It manages the checkpointing and holds a member 'state' that contains
    information about the current state of the training"""

    def __init__(self) -> None:
        self.model = ModelClass()

        count_parameters(self.model)

        self.optim = getattr(torch.optim, conf.optimizer.name)(
            self.model.parameters(),
            **(
                conf.optimizer.parameters
                if conf.optimizer.parameters is not None
                else {}
            ),
        )
        self.lossf = getattr(torch.nn, conf.loss.name)().to(device)

        self.state = OmegaConf.create(
            {
                "epoch": 0,
                "processed_events": 0,
                "ibatch": 0,
                "grad_step": 0,
                "val_losses": [],
            }
        )
        self.__load_checkpoint()
        self.model = self.model.float().to(device)
        # Hack to move the optimizer parameters to the correct device
        # https://github.com/pytorch/pytorch/issues/8741
        self.optim.load_state_dict(self.optim.state_dict())

    def __load_checkpoint(self):
        if not os.path.isfile(conf.path.checkpoint):
            logger.warn("Proceeding without loading checkpoint.")
            return

        checkpoint = torch.load(conf.path.checkpoint, map_location=device)

        assert not contains_nans(checkpoint["model"])[0]
        assert not contains_nans(checkpoint["best_model"])[0]

        self.state = checkpoint["state"]
        self.model.load_state_dict(checkpoint["model"])
        self.optim.load_state_dict(checkpoint["optim"])
        self.best_model_state = checkpoint["best_model"]

        logger.warn(
            "Loading model from checkpoint at"
            + f" epoch {self.state['epoch']}"
            + f" batch {self.state['ibatch']}"
            + f" grad_step {self.state['grad_step']}."
        )

    def select_best_model(self):
        self.model.load_state_dict(self.best_model_state)
        self.model = self.model.float().to(device)

    def save_checkpoint(self):
        self.__push_to_old(conf.path.checkpoint, conf.path.checkpoint_old)

        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "state": self.state,
                "best_model": self.best_model_state,
            },
            conf.path.checkpoint,
        )

    def __push_to_old(self, path_new, path_old):
        if os.path.isfile(path_new):
            if os.path.isfile(path_old):
                os.remove(path_old)
            os.rename(path_new, path_old)


model_holder = ModelHolder()
