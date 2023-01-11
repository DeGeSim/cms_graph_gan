from pathlib import Path
from typing import List

import comet_ml
import wandb
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from fgsim.utils.oc_utils import dict_to_kv

from .experiment_organizer import ExperimentOrganizer

comet_conf = OmegaConf.load(Path("~/fgsim/fgsim/comet.yaml").expanduser())
comet_api = comet_ml.API(comet_conf.api_key)


exp_orga_comet = ExperimentOrganizer("comet")
exp_orga_wandb = ExperimentOrganizer("wandb")


def search_experiement_by_name(exp_hash: str) -> List[comet_ml.APIExperiment]:
    workspace = comet_conf.workspace
    exps_whash = []
    for project in comet_api.get(workspace):
        for exp in comet_api.get(workspace, project):
            if exp.name == exp_hash:
                exps_whash.append(exp)
    return exps_whash


def experiment_from_hash(hash) -> comet_ml.ExistingExperiment:
    return comet_ml.ExistingExperiment(
        previous_experiment=exp_orga_comet[hash],
        **comet_conf,
        log_code=True,
        log_graph=True,
        parse_args=True,
        log_env_details=True,
        log_git_metadata=True,
        log_git_patch=True,
        log_env_gpu=True,
        log_env_cpu=True,
        log_env_host=True,
        auto_output_logging=False,
    )


def get_experiment(state: DictConfig) -> comet_ml.ExistingExperiment:
    from fgsim.config import conf

    experiment = experiment_from_hash(conf.hash)

    experiment.set_step(state["grad_step"])
    experiment.set_epoch(state["epoch"])
    return experiment


def setup_experiment() -> None:
    from fgsim.config import conf

    # comet ml
    """Generates a new experiment."""
    if conf.hash in exp_orga_comet.keys():
        if conf.ray:
            return
        raise Exception("Experiment exists")
    project_name = (
        conf.comet_project_name
        if "comet_project_name" in conf
        else comet_conf.project_name
    )
    new_api_exp = comet_api._create_experiment(
        workspace=comet_conf.workspace, project_name=project_name
    )

    # Format the hyperparameter for comet
    from fgsim.config import hyperparameters

    assert len(hyperparameters) > 0
    hyperparameters_keyval_list = dict(dict_to_kv(hyperparameters))
    hyperparameters_keyval_list["hash"] = conf["hash"]
    hyperparameters_keyval_list["loader_hash"] = conf["loader_hash"]
    new_api_exp.log_parameters(hyperparameters_keyval_list)
    new_api_exp.log_other("name", conf["hash"])
    tags_list = list(set(conf.tag.split("_")))
    new_api_exp.add_tags(tags_list)
    exp_orga_comet[conf["hash"]] = new_api_exp.id

    # wandb
    run = wandb.init(
        project=conf.comet_project_name,
        name=conf["hash"],
        tags=tags_list,
        config=hyperparameters_keyval_list,
        dir=conf.path.run_path,
        resume=False,
    )
    exp_orga_wandb[conf["hash"]] = run.id
