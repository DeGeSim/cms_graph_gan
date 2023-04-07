import ray
from hyperpars import hyperpars
from omegaconf import OmegaConf
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.stopper import ExperimentPlateauStopper
from runconf import process_exp_config, rayconf, run_name
from trainable import MyTrainable

import fgsim.config


def trial_name_id(trial):
    comconf, _ = fgsim.config.compute_conf(
        fgsim.config.defaultconf,
        rayconf,
        process_exp_config(OmegaConf.create(trial.config)),
    )
    return comconf.hash


def trial_dirname_creator(trial):
    return str(trial)


local = False
# local = True
print(f"start tune in local model: {local}")

if local:
    ray.init(local_mode=True)
else:
    ray.init("auto")
try:
    tuner = ray.tune.Tuner.restore(
        path=f"~/fgsim/wd/ray/{run_name}",
        trainable=tune.with_resources(MyTrainable, {"gpu": 1}),
        param_space=hyperpars,
    )
except Exception:
    tuner = tune.Tuner(
        tune.with_resources(MyTrainable, {"gpu": 1}),
        param_space=hyperpars,
        run_config=air.RunConfig(
            callbacks=[WandbLoggerCallback(project=run_name)],
            log_to_file=True,
            name=run_name,
            local_dir="~/fgsim/wd/ray/",
            stop=ExperimentPlateauStopper(metric="fpnd"),
            failure_config=air.FailureConfig(fail_fast="raise" if local else False),
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="fpnd",
                checkpoint_frequency=5,
                checkpoint_at_end=True,
                checkpoint_score_order="min",
            ),
            sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
        ),
        tune_config=tune.TuneConfig(
            mode="min",
            metric="fpnd",
            scheduler=MedianStoppingRule(
                time_attr="training_iteration", grace_period=110.0
            ),
            search_alg=HyperOptSearch(),
            num_samples=-1,
            trial_name_creator=trial_name_id,
            trial_dirname_creator=trial_dirname_creator,
            chdir_to_trial_dir=False,
        ),
    )
# try:
#     tuner = tuner.restore(f"~/fgsim/wd/ray/{run_name}")
# except:
#     pass

analysis = tuner.fit()

best_trial = analysis.best_trial  # Get best trial
best_config = analysis.best_config  # Get best trial's hyperparameters
best_logdir = analysis.best_logdir  # Get best trial's logdir
best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
best_result = analysis.best_result  # Get best trial's last results
best_result_df = analysis.best_result_df  # Get best result as pandas dataframe

best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="accuracy")
print(best_checkpoint)
