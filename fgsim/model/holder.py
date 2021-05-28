import os

import torch

from ..config import conf, device
from ..utils.count_parameters import count_parameters
from ..utils.logger import logger
from .model import Net


class modelHolder:
    def __init__(self) -> None:
        # self.train_data = torch.tensor(mapper.map_events(eventarr))

        self.metrics = {
            "loss": [],
            "acc": [],
            "losses_g": [],
            "losses_d": [],
            "memory": [],
            "epoch": 0,
            "grad_step": 0,
            "batch": 0,
        }
        self.checkpoint_path = f"wd/{conf.tag}/checkpoint.torch"
        self.checkpoint_old_path = f"wd/{conf.tag}/checkpoint_old.torch"

        # self.discriminator = Discriminator().to(device)
        # self.generator = Generator(conf.model.gan.nz).to(device)
        self.model = Net().to(device)

        # count_parameters(self.generator)
        # count_parameters(self.discriminator)
        count_parameters(self.model)
        # optimizers
        # self.optim_g = optim.Adam(self.generator.parameters(), lr=conf.model.gan.lr)
        # self.optim_d = optim.Adam(self.discriminator.parameters(), lr=conf.model.gan.lr)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=0.01, weight_decay=5e-4
        )
        # loss function
        # self.lossf = torch.nn.BCELoss().to(device)
        self.lossf = torch.nn.MSELoss().to(device)
        self.load_checkpoint()

    def load_checkpoint(self):
        if not os.path.isfile(self.checkpoint_path) or conf["dump_model"]:
            return
        checkpoint = torch.load(self.checkpoint_path)

        # self.discriminator.load_state_dict(checkpoint["discriminator"])
        # self.discriminator.eval()
        # self.generator.load_state_dict(checkpoint["generator"])
        # self.generator.eval()
        # self.optim_d.load_state_dict(checkpoint["optim_d"])
        # self.optim_g.load_state_dict(checkpoint["optim_g"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.optim.load_state_dict(checkpoint["optim"])
        self.metrics = checkpoint["metrics"]
        logger.warn(
            f"Loading model from checkpoint at"
            + f" epoch {self.metrics['epoch']}"
            + f" batch {self.metrics['batch']}"
            + f" grad_step {self.metrics['grad_step']}."
        )

    def save_model(self):
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
            #     "metrics": self.metrics,
            # },
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "metrics": self.metrics,
            },
            self.checkpoint_path,
        )
        logger.warn(
            f"Saved model to checkpoint at epoch {self.metrics['epoch']} / gradient step {self.metrics['grad_step']}."
        )


model_holder = modelHolder()
