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
        self.state_path = f"wd/{conf.tag}/state.yaml"
        self.state_old_path = f"wd/{conf.tag}/state_old.yaml"
        self.checkpoint_path = f"wd/{conf.tag}/checkpoint.torch"
        self.checkpoint_old_path = f"wd/{conf.tag}/checkpoint_old.torch"
        self.best_model_path = f"wd/{conf.tag}/best_model.torch"

        # self.discriminator = Discriminator().to(device)
        # self.generator = Generator(conf.model.gan.nz).to(device)
        self.model = ModelClass().to(device)

        # count_parameters(self.generator)
        # count_parameters(self.discriminator)
        count_parameters(self.model)
        # optimizers
        # self.optim_g = optim.Adam(self.generator.parameters(), lr=conf.model.gan.lr)
        # self.optim_d = optim.Adam(self.discriminator.parameters(), lr=conf.model.gan.lr)

        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=conf.model.lr, weight_decay=1e-5
        )
        # loss function
        # self.lossf = torch.nn.BCELoss().to(device)
        self.lossf = torch.nn.MSELoss().to(device)
        self.load_checkpoint()

        # state when training is started new
        self.state = OmegaConf.merge(
            OmegaConf.create(
                {
                    "epoch": 0,
                    "processed_events": 0,
                    "ibatch": 0,
                    "grad_step": 0,
                    "val_losses": [],
                }
            ),
            self.state,
        )

    def load_checkpoint(self):
        if (
            not os.path.isfile(self.checkpoint_path)
            or not os.path.isfile(self.state_path)
            or conf["dump_model"]
        ):
            logger.warn("Dumping model.")
            self.state = OmegaConf.create()
            return
        self.state = OmegaConf.load(self.state_path)
        checkpoint = torch.load(self.checkpoint_path)

        # self.discriminator.load_state_dict(checkpoint["discriminator"])
        # self.discriminator.eval()
        # self.generator.load_state_dict(checkpoint["generator"])
        # self.generator.eval()
        # self.optim_d.load_state_dict(checkpoint["optim_d"])
        # self.optim_g.load_state_dict(checkpoint["optim_g"])

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
        if os.path.isfile(self.checkpoint_path):
            if os.path.isfile(self.checkpoint_old_path):
                os.remove(self.checkpoint_old_path)
            os.rename(self.checkpoint_path, self.checkpoint_old_path)

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
            self.checkpoint_path,
        )
        if os.path.isfile(self.state_path):
            if os.path.isfile(self.state_old_path):
                os.remove(self.state_old_path)
            os.rename(self.state_path, self.state_old_path)
        OmegaConf.save(self.state, self.state_path)
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
            self.best_model_path,
        )


model_holder = modelHolder()
