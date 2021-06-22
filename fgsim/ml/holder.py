import importlib
import os

import torch
from omegaconf import OmegaConf

from ..config import conf, device
from ..utils.count_parameters import count_parameters
from ..utils.logger import logger

# Import the specified model
ModelClass = importlib.import_module(
    f"..models.{conf.model.name}", "fgsim.models"
).ModelClass


class ModelHolder:
    def __init__(self) -> None:
        self.model = ModelClass().to(device)

        count_parameters(self.model)

        self.optim = getattr(torch.optim, conf.optimizer.name)(
            self.model.parameters(), **conf.optimizer.parameters
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
        self.load_models()

    def save_models(self):
        self.__save_state()
        self.__save_checkpoint()
        self.__save_best_model()

    def load_models(self):
        if (
            not os.path.isfile(conf.path.checkpoint)
            or not os.path.isfile(conf.path.state)
            or not os.path.isfile(conf.path.best_model)
            or conf["dump_model"]
        ):
            logger.warn("Proceeding without checkpoint.")
            return
        self.__load_state()
        self.__load_checkpoint()
        self.__load_best_model()

        logger.warn(
            "Loading model from checkpoint at"
            + f" epoch {self.state['epoch']}"
            + f" batch {self.state['ibatch']}"
            + f" grad_step {self.state['grad_step']}."
        )

    def __load_state(self):
        self.state = OmegaConf.load(conf.path.state)

    def __load_checkpoint(self):
        checkpoint = torch.load(conf.path.checkpoint)

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.optim.load_state_dict(checkpoint["optim"])

    def __load_best_model(self):
        checkpoint = torch.load(conf.path.best_model)
        self.best_model_state = checkpoint["model"]

    def __save_checkpoint(self):
        # move the old checkpoint
        if os.path.isfile(conf.path.checkpoint):
            if os.path.isfile(conf.path.checkpoint_old):
                os.remove(conf.path.checkpoint_old)
            os.rename(conf.path.checkpoint, conf.path.checkpoint_old)

        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
            },
            conf.path.checkpoint,
        )

    def __save_state(self):
        if os.path.isfile(conf.path.state):
            if os.path.isfile(conf.path.state_old):
                os.remove(conf.path.state_old)
            os.rename(conf.path.state, conf.path.state_old)
        OmegaConf.save(self.state, conf.path.state)
        logger.warn(
            f"Saved model to checkpoint at epoch {self.state['epoch']} /"
            + f" gradient step {self.state['grad_step']}."
        )

    def __save_best_model(self):
        torch.save(
            {
                "model": self.best_model_state,
            },
            conf.path.best_model,
        )


model_holder = ModelHolder()
