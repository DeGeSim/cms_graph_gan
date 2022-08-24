# %%
import os
from pathlib import Path
from typing import List

from omegaconf import OmegaConf
from ray.tune import ExperimentAnalysis
from ray.tune.trial import Trial

import fgsim.config

analysis = ExperimentAnalysis(
    "~/fgsim/wd/ray/jetnet-deeptree", default_metric="fpnd", default_mode="min"
)
# %%
best_trials: List[Trial] = [
    analysis.get_best_trial(metric, "min", scope="all")
    for metric in ["fpnd", "w1m", "fpnd"]
]
# %%


# %%
for trial in best_trials:
    config = OmegaConf.create(trial.config)
    config["path"] = {"run_path": trial.logdir}
    config["command"] = "train"
    config["debug"] = False
    config["comet_project_name"] = Path(trial.logdir).parts[-2]
    config["ray"] = True
    fgsim.config.conf, _ = fgsim.config.compute_conf(
        fgsim.config.defaultconf, config
    )
    best_checkpoint_path = Path(analysis.get_best_checkpoint(trial).local_path)
    ckp_source = best_checkpoint_path / "cp.pth"
    state_source = best_checkpoint_path / "state.pth"
    os.symlink(ckp_source, fgsim.config.conf.path.checkpoint)
    os.symlink(state_source, fgsim.config.conf.path.state)
