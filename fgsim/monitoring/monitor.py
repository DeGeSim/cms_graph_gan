from typing import List

import comet_ml
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.tensorboard.writer import SummaryWriter

from fgsim.config import conf
from fgsim.utils.oc_utils import dict_to_kv

comet_conf = OmegaConf.load("fgsim/comet.yaml")
api = comet_ml.API(comet_conf.api_key)
project_name = (
    conf.comet_project_name
    if "comet_project_name" in conf
    else comet_conf.project_name
)


def get_exps_with_hash(hash: str) -> List[comet_ml.APIExperiment]:
    experiments = [
        exp
        for exp in api.get(
            workspace=comet_conf.workspace,
            project_name=project_name,
        )
        if exp.get_parameters_summary("hash") != []
    ]
    qres = [
        exp
        for exp in experiments
        if exp.get_parameters_summary("hash")["valueCurrent"] == hash
    ]
    return qres


def experiment_from_key(key) -> comet_ml.ExistingExperiment:
    experiment = comet_ml.ExistingExperiment(
        previous_experiment=key,
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
    )
    return experiment


def experiment_from_hash(hash) -> comet_ml.ExistingExperiment:
    qres = get_exps_with_hash(hash)
    if len(qres) == 0:
        raise ValueError("Experiment does not exist in comet.ml")
    elif len(qres) > 1:
        raise ValueError("Experiment exist multiple times in comet.ml")

    exp_key = qres[0].id

    experiment = experiment_from_key(exp_key)
    return experiment


def get_experiment(state: DictConfig) -> comet_ml.ExistingExperiment:
    experiment = experiment_from_hash(conf.hash)

    experiment.set_step(state["grad_step"])
    experiment.set_epoch(state["epoch"])
    return experiment


def setup_experiment() -> None:
    """Generates a new experiment."""
    if len(get_exps_with_hash(conf.hash)):
        raise Exception("Experiment exists")

    new_api_exp = api._create_experiment(
        workspace=comet_conf.workspace, project_name=project_name
    )

    # Format the hyperparameter for comet
    from fgsim.config import hyperparameters

    hyperparameters_keyval_list = dict(dict_to_kv(hyperparameters))
    hyperparameters_keyval_list["hash"] = conf["hash"]
    hyperparameters_keyval_list["loader_hash"] = conf["loader_hash"]
    new_api_exp.log_parameters(hyperparameters_keyval_list)
    new_api_exp.add_tags(list(set(conf.tag.split("_"))))

    # exp_key = new_api_exp.id

    # experiment = experiment_from_key(exp_key)
    # experiment.log_code(Path(conf.path.run_path) / "fgsim")


def get_writer():
    return SummaryWriter(conf.path.tensorboard)
