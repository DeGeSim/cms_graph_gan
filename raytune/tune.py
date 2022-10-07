from pathlib import Path

import numpy as np
import ray
from hyperpars import hyperpars
from omegaconf import OmegaConf
from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.stopper import ExperimentPlateauStopper
from ray.tune.suggest.hyperopt import HyperOptSearch

import fgsim.config

run_name = "jetnet-deeptree4"
rayconf = OmegaConf.load(Path(f"~/fgsim/wd/ray/{run_name}/conf.yaml").expanduser())


def process_tree_conf(exp_config):
    # manipulate the config for the tree
    wide = exp_config["tree_width"] == "wide"
    root_size = exp_config["root_node_size"]
    exp_config["tree"] = {}
    if wide:
        exp_config.tree["branches"] = [3, 10]
        scaling = np.power(root_size / 3.0, 1 / 2)
        exp_config.tree["features"] = [root_size, int(root_size / scaling), 3]

    else:
        exp_config.tree["branches"] = [2, 3, 5]
        scaling = np.power(root_size / 3.0, 1 / 3)
        exp_config.tree["features"] = [
            root_size,
            int(root_size / scaling),
            int(root_size / scaling**2),
            3,
        ]

    del exp_config["tree_width"]
    del exp_config["root_node_size"]
    return exp_config


class Trainable(tune.Trainable):
    def setup(self, exp_config):
        self.exp_config = process_tree_conf(OmegaConf.create(exp_config))

        OmegaConf.save(self.exp_config, Path(self.logdir) / "exp.yaml")
        self.exp_config["path"] = {"run_path": self._logdir}
        self.exp_config["command"] = "train"
        self.exp_config["debug"] = False
        self.exp_config["comet_project_name"] = Path(self.logdir).parts[-2]
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
        return {k: v[-1] for k, v in self.holder.history["val_metrics"].items()}

    def cleanup(self):
        self.trainer.loader.qfseq.stop()

    def save_checkpoint(self, tmp_checkpoint_dir):
        return self.holder.save_ray_checkpoint(tmp_checkpoint_dir)

    def load_checkpoint(self, checkpoint):
        return self.holder.load_ray_checkpoint(checkpoint)


def trial_name_id(trial):
    comconf, _ = fgsim.config.compute_conf(
        fgsim.config.defaultconf, process_tree_conf(OmegaConf.create(trial.config))
    )
    return comconf.hash


def trial_dirname_creator(trial):
    return trial


local = False
print(f"start tune in local model: {local}")

if local:
    ray.init(local_mode=True)
else:
    ray.init("auto")
analysis = tune.run(
    Trainable,
    config=hyperpars,
    mode="min",
    metric="fpnd",
    scheduler=MedianStoppingRule(
        time_attr="training_iteration", grace_period=110.0
    ),
    search_alg=HyperOptSearch(),
    num_samples=-1,
    keep_checkpoints_num=2,
    checkpoint_score_attr="fpnd",
    checkpoint_freq=5,
    checkpoint_at_end=True,
    log_to_file=True,
    resources_per_trial={"cpu": 15, "gpu": 1},
    fail_fast="raise" if local else False,
    raise_on_failed_trial=not local,
    name=run_name,
    local_dir="~/fgsim/wd/ray/",
    stop=ExperimentPlateauStopper(metric="fpnd"),
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
