from pathlib import Path

import ray
from hyperpars import hyperpars
from omegaconf import OmegaConf
from ray import tune
from ray.tune.schedulers import MedianStoppingRule

import fgsim.config

# from ray.tune.suggest.bayesopt import BayesOptSearch
# from ray.tune.suggest.hyperopt import HyperOptSearch


class Trainable(tune.Trainable):
    def setup(self, exp_config):
        self.exp_config = OmegaConf.create(exp_config)
        OmegaConf.save(self.exp_config, Path(self.logdir) / "config.yaml")
        self.exp_config["path"] = {"run_path": self._logdir}
        self.exp_config["command"] = "train"
        self.exp_config["debug"] = False
        self.exp_config["comet_project_name"] = Path(self.logdir).parts[-2]
        self.exp_config["ray"] = True
        fgsim.config.conf, _ = fgsim.config.compute_conf(
            fgsim.config.defaultconf, self.exp_config
        )
        fgsim.config.device = fgsim.config.get_device()
        from fgsim.monitoring.logger import init_logger

        init_logger()
        from fgsim.commands.training import Trainer
        from fgsim.ml.early_stopping import early_stopping
        from fgsim.ml.holder import Holder

        self.early_stoppingf = early_stopping
        # from fgsim.monitoring.monitor import setup_experiment
        # setup_experiment()
        self.holder = Holder()
        self.trainer = Trainer(self.holder)

    def step(self):
        if (
            self.early_stoppingf(self.holder.history)
            or self.holder.state.epoch > fgsim.config.conf.training.max_epochs
        ):
            self.stop()
        self.trainer.train_epoch()
        return {k: v[-1] for k, v in self.holder.history["val_metrics"].items()}

    def cleanup(self):
        self.trainer.loader.qfseq.stop()

    def save_checkpoint(self, tmp_checkpoint_dir):
        return self.holder.save_ray_checkpoint(Path(tmp_checkpoint_dir))

    def load_checkpoint(self, checkpoint):
        return self.holder.load_ray_checkpoint(Path(checkpoint))


def trial_name_id(trial):
    comconf, _ = fgsim.config.compute_conf(
        fgsim.config.defaultconf, OmegaConf.create(trial.config)
    )
    return comconf.hash


def trial_dirname_creator(trial):
    return trial


# ray.init(local_mode=True)
ray.init()
analysis = tune.run(
    Trainable,
    config=hyperpars,
    mode="min",
    metric="fpnd",
    scheduler=MedianStoppingRule(
        time_attr="training_iteration", grace_period=110.0
    ),
    keep_checkpoints_num=2,
    checkpoint_score_attr="fpnd",
    checkpoint_freq=5,
    checkpoint_at_end=True,
    log_to_file=True,
    resources_per_trial={"cpu": 15, "gpu": 1},
    fail_fast=False,
    raise_on_failed_trial=False,
    name="jetnet-deeptree",
    local_dir="~/fgsim/wd/ray/",
    trial_name_creator=trial_name_id,
    trial_dirname_creator=trial_name_id,
    resume="AUTO",
    sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
)

best_trial = analysis.best_trial  # Get best trial
best_config = analysis.best_config  # Get best trial's hyperparameters
best_logdir = analysis.best_logdir  # Get best trial's logdir
best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
best_result = analysis.best_result  # Get best trial's last results
best_result_df = analysis.best_result_df  # Get best result as pandas dataframe

best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="accuracy")
print(best_checkpoint)
