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


class modelHolder:
    def __init__(self) -> None:
        # self.discriminator = Discriminator().to(device)
        # self.generator = Generator(conf.model.gan.nz).to(device)
        self.model = ModelClass().to(device)

        # count_parameters(self.generator)
        # count_parameters(self.discriminator)
        count_parameters(self.model)
        # optimizers
        # self.optim_g = optim.Adam(self.generator.parameters(), lr=conf.model.gan.lr)
        # self.optim_d = optim.Adam(self.discriminator.parameters(), lr=conf.model.gan.lr)

        self.optim = getattr(torch.optim, conf.optimizer.name)(
            self.model.parameters(), **conf.optimizer.parameters
        )

        # loss function
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

    def load_checkpoint(self):
        if (
            not os.path.isfile(conf.path.checkpoint)
            or not os.path.isfile(conf.path.state)
            or conf["dump_model"]
        ):
            logger.warn("Dumping model.")
            return
        self.state = OmegaConf.load(conf.path.state)
        checkpoint = torch.load(conf.path.checkpoint)

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.best_state_model = self.model.state_dict()

        self.optim.load_state_dict(checkpoint["optim"])
        self.best_state_optim = self.optim.state_dict()

        logger.warn(
            "Loading model from checkpoint at"
            + f" epoch {self.state['epoch']}"
            + f" batch {self.state['ibatch']}"
            + f" grad_step {self.state['grad_step']}."
        )

    def save_checkpoint(self):
        # move the old checkpoint
        if os.path.isfile(conf.path.checkpoint):
            if os.path.isfile(conf.path.checkpoint_old):
                os.remove(conf.path.checkpoint_old)
            os.rename(conf.path.checkpoint, conf.path.checkpoint_old)

        torch.save(
            # {
            #     "discriminator": self.discriminator.state_dict(),
            #     "generator": self.generator.state_dict(),
            #     "optim_d": self.optim_d.state_dict(),
            #     "optim_g": self.optim_g.state_dict(),
            #     "metrics": self.state,
            # },
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
            },
            conf.path.checkpoint,
        )
        if os.path.isfile(conf.path.state):
            if os.path.isfile(conf.path.state_old):
                os.remove(conf.path.state_old)
            os.rename(conf.path.state, conf.path.state_old)
        OmegaConf.save(self.state, conf.path.state)
        logger.warn(
            f"Saved model to checkpoint at epoch {self.state['epoch']} /"
            + f" gradient step {self.state['grad_step']}."
        )

    def save_best_model(self):
        torch.save(
            {
                "model": self.best_state_model,
                "optim": self.best_state_optim,
            },
            conf.path.best_model,
        )

    def load_best_model(self):
        # checkpoint = torch.load(conf.path.best_model)
        checkpoint = torch.load(conf.path.best_model)

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()


model_holder = modelHolder()
