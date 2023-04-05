import ray
from hyperpars import hyperpars
from omegaconf import OmegaConf
from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.stopper import ExperimentPlateauStopper
from ray.tune.suggest.hyperopt import HyperOptSearch

import fgsim.config
from raytune.runconf import process_tree_conf, rayconf, run_name
from raytune.trainable import Trainable


def trial_name_id(trial):
    comconf, _ = fgsim.config.compute_conf(
        fgsim.config.defaultconf,
        rayconf,
        process_tree_conf(OmegaConf.create(trial.config)),
    )
    return comconf.hash


def trial_dirname_creator(trial):
    return trial


# local = False
local = True
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
    # sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
)

best_trial = analysis.best_trial  # Get best trial
best_config = analysis.best_config  # Get best trial's hyperparameters
best_logdir = analysis.best_logdir  # Get best trial's logdir
best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
best_result = analysis.best_result  # Get best trial's last results
best_result_df = analysis.best_result_df  # Get best result as pandas dataframe

best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="accuracy")
print(best_checkpoint)
