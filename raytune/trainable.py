from pathlib import Path

from omegaconf import OmegaConf
from ray import tune

import fgsim.config

from .runconf import process_tree_conf, rayconf


class Trainable(tune.Trainable):
    def setup(self, exp_config):
        self.exp_config = process_tree_conf(OmegaConf.create(exp_config))

        OmegaConf.save(self.exp_config, Path(self.logdir) / "exp.yaml")
        self.exp_config["path"] = {"run_path": self._logdir}
        self.exp_config["command"] = "train"
        self.exp_config["debug"] = False
        self.exp_config["project_name"] = Path(self.logdir).parts[-2]
        self.exp_config["ray"] = True
        fgsim.config.conf, _ = fgsim.config.compute_conf(
            fgsim.config.defaultconf, rayconf, self.exp_config
        )
        assert fgsim.config.conf.hash in self.logdir
        assert self.trial_id == fgsim.config.conf.hash
        OmegaConf.save(fgsim.config.conf, Path(self.logdir) / "conf.yaml")
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

    def step(self):
        if self.holder.state.epoch > fgsim.config.conf.training.max_epochs:
            self.stop()
        self.trainer.train_epoch()
        if self.early_stoppingf(self.holder):
            self.stop()
        return {k: v[-1] for k, v in self.holder.history["val"].items()}

    def cleanup(self):
        self.trainer.loader.qfseq.stop()

    def save_checkpoint(self, tmp_checkpoint_dir):
        return self.holder.save_ray_checkpoint(tmp_checkpoint_dir)

    def load_checkpoint(self, checkpoint):
        return self.holder.load_ray_checkpoint(checkpoint)
