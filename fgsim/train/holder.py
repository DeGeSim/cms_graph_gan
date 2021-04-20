import os

import torch

from ..config import conf, device
# from ..geo.mapper import Geomapper
from ..utils.count_parameters import count_parameters
from .model import Net


class modelHolder:
    def __init__(self) -> None:
        # self.train_data = torch.tensor(mapper.map_events(eventarr))

        self.metrics = {"losses_g": [], "losses_d": [], "memory": [], "epoch": 0}
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
        self.lossf = torch.nn.MSELoss()
        self.load_checkpoint()

    def load_checkpoint(self):
        if not os.path.isfile(self.checkpoint_path) or conf["dump_model"]:
            return
        checkpoint = torch.load(self.checkpoint_path)

        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.discriminator.eval()
        self.generator.load_state_dict(checkpoint["generator"])
        self.generator.eval()
        self.optim_d.load_state_dict(checkpoint["optim_d"])
        self.optim_g.load_state_dict(checkpoint["optim_g"])
        self.metrics = checkpoint["metrics"]

    def save_model(self):
        # move the old checkpoint
        if os.path.isfile(self.checkpoint_path):
            if os.path.isfile(self.checkpoint_old_path):
                os.remove(self.checkpoint_old_path)
            os.rename(self.checkpoint_path, self.checkpoint_old_path)
        torch.save(
            {
                "discriminator": self.discriminator.state_dict(),
                "generator": self.generator.state_dict(),
                "optim_d": self.optim_d.state_dict(),
                "optim_g": self.optim_g.state_dict(),
                "metrics": self.metrics,
            },
            self.checkpoint_path,
        )

    # def run_training(self):
    #     from .train import training_procedure

    #     self.generator, self.discriminator = training_procedure(self)

    #     logger.info("Training finished")
    #     torch.save(self.generator.state_dict(), "output/generator.pth")
    #     logger.info("Model saved")


model_holder = modelHolder()
