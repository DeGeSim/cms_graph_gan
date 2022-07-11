from typing import List

import comet_ml
import yaml
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.utils.tensorboard.writer import SummaryWriter

from fgsim.utils.oc_utils import dict_to_kv

comet_conf = OmegaConf.load("fgsim/comet.yaml")
api = comet_ml.API(comet_conf.api_key)


class ExperimentOrganizer:
    def __init__(self) -> None:
        self.fn = "wd/hash2exp.yaml"
        with open(self.fn, "r") as f:
            self.d = yaml.load(f, Loader=yaml.SafeLoader)

    def __getitem__(self, h: str) -> str:
        return self.d[h]

    def save(self):
        with open(self.fn, "w") as f:
            yaml.dump(self.d, f, Dumper=yaml.SafeDumper)

    def recreate(self):
        workspace = comet_conf.workspace
        experiments = []
        for project in api.get_projects(workspace):
            experiments = experiments + api.get(workspace, project)
        self.d = {}
        for exp in experiments:
            self.d[exp.name] = exp.id
        self.save()


exp_orga = ExperimentOrganizer()


def api_experiment_from_hash(hash: str) -> comet_ml.APIExperiment:
    return api.get_experiment_by_key(exp_orga[hash])


def search_experiement_by_name(exp_hash: str) -> List[comet_ml.APIExperiment]:
    workspace = comet_conf.workspace
    exps_whash = []
    for project in api.get(workspace):
        for exp in api.get(workspace, project):
            if exp.name == exp_hash:
                exps_whash.append(exp)
    return exps_whash


def experiment_from_hash(hash) -> comet_ml.ExistingExperiment:
    return comet_ml.ExistingExperiment(
        previous_experiment=exp_orga[hash],
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

    """Generates a new experiment."""
    if conf.hash in exp_orga.d:
        raise Exception("Experiment exists")
    project_name = (
        conf.comet_project_name
        if "comet_project_name" in conf
        else comet_conf.project_name
    )
    new_api_exp = api._create_experiment(
        workspace=comet_conf.workspace, project_name=project_name
    )

    # Format the hyperparameter for comet
    from fgsim.config import hyperparameters

    hyperparameters_keyval_list = dict(dict_to_kv(hyperparameters))
    hyperparameters_keyval_list["hash"] = conf["hash"]
    hyperparameters_keyval_list["loader_hash"] = conf["loader_hash"]
    new_api_exp.log_parameters(hyperparameters_keyval_list)
    new_api_exp.log_other("name", conf["hash"])
    new_api_exp.add_tags(list(set(conf.tag.split("_"))))
    exp_orga.d[conf["hash"]] = new_api_exp.id
    exp_orga.save()


def get_writer():
    from fgsim.config import conf

    return SummaryWriter(conf.path.tensorboard)
