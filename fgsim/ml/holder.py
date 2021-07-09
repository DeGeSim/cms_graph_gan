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
        self.__load_models()
        self.model = self.model.float().to(device)

    def __load_models(self):
        if (
            not os.path.isfile(conf.path.state)
            or not os.path.isfile(conf.path.checkpoint)
            or not os.path.isfile(conf.path.best_model)
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
        checkpoint = torch.load(conf.path.checkpoint, map_location=device)
        assert not contains_nans(checkpoint)[0]

        self.model.load_state_dict(checkpoint["model"])
        if device.type == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        self.optim.load_state_dict(checkpoint["optim"])

    def __load_best_model(self):
        checkpoint = torch.load(conf.path.best_model, map_location=device)
        assert not contains_nans(checkpoint)[0]
        self.best_model_state = checkpoint["model"]

    def save_models(self):
        self.__save_state()
        self.__save_checkpoint()
        self.__save_best_model()

    def __save_checkpoint(self):
        self.__push_to_old(conf.path.checkpoint, conf.path.checkpoint_old)

        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
            },
            conf.path.checkpoint,
        )

    def __save_state(self):
        self.__push_to_old(conf.path.state, conf.path.state_old)
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

    def __push_to_old(self, path_new, path_old):
        if os.path.isfile(path_new):
            if os.path.isfile(path_old):
                os.remove(path_old)
            os.rename(path_new, path_old)


model_holder = ModelHolder()
