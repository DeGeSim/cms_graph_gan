import os
from pathlib import Path

from hyperpars import project_name
from omegaconf import OmegaConf

import fgsim.config
import wandb

os.chdir(Path("~").expanduser())


class MyTrainable:
    def __init__(self):
        fgsim.config.device = fgsim.config.get_device()
        from fgsim.monitoring.logger import init_logger

        init_logger()
        from fgsim.commands.training import Trainer
        from fgsim.config import device
        from fgsim.ml.early_stopping import early_stopping
        from fgsim.ml.holder import Holder

        self.early_stoppingf = early_stopping
        # from fgsim.monitoring.monitor import setup_experiment
        # setup_experiment()
        self.holder = Holder(device)
        self.trainer = Trainer(self.holder)


def run_hyperpar_cfg():
    wandb.init()
    cfg = dict(wandb.config)
    fgsim.config.defaultconf["command"] = "train"
    fgsim.config.defaultconf["project_name"] = project_name
    fgsim.config.defaultconf["ray"] = True
    fgsim.config.defaultconf["debug"] = False
    fgsim.config.conf, hyperparameters = fgsim.config.compute_conf(
        fgsim.config.defaultconf, OmegaConf.create(cfg)
    )
    conf = fgsim.config.conf

    # If the experiment has been setup, exit directly
    if Path(conf.path.run_path).is_dir():
        return conf.hash

    os.makedirs(conf.path.run_path, exist_ok=True)

    OmegaConf.save(conf, Path(conf.path.run_path) / "conf.yaml")
    OmegaConf.save(
        hyperparameters, Path(conf.path.run_path) / "hyperparameters.yaml"
    )
    OmegaConf.save(cfg, Path(conf.path.run_path) / "exp.yaml")
    trainable = MyTrainable()
    trainable.trainer.training_loop()


with open(f"fgsim/wd/{project_name}/sweepid", "r") as f:
    sweep_id = f.read()
wandb.agent(sweep_id, function=run_hyperpar_cfg, project=project_name)
