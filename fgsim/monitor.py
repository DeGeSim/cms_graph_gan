import comet_ml
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .config import conf, hyperparameters
from .utils.logger import logger


def dict_to_kv(o, keystr=""):
    """Converts a nested dict {"a":"foo", "b": {"foo":"bar"}} to \
    [("a","foo"),("b.foo","bar")]."""
    if hasattr(o, "keys"):
        outL = []
        for k in o.keys():
            elemres = dict_to_kv(o[k], keystr + str(k) + ".")
            if (
                len(elemres) == 2
                and type(elemres[0]) == str
                and type(elemres[1]) == str
            ):
                outL.append(elemres)
            else:
                for e in elemres:
                    outL.append(e)
        return outL
    elif hasattr(o, "__str__"):

        return (keystr.strip("."), str(o))
    else:
        raise ValueError


def get_experiment():
    """Tries to find for an existing experiment with the given hash and \
 -- if unsuccessfull -- generates a new one."""
    comet_conf = OmegaConf.load("fgsim/comet.yaml")
    api = comet_ml.API(comet_conf.api_key)
    project_name = (
        conf.comet_project_name
        if "comet_project_name" in conf
        else comet_conf.project_name
    )

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
        if exp.get_parameters_summary("hash")["valueCurrent"] == conf.hash
    ]
    # No experiment with the given hash:
    if len(qres) == 0:
        if conf.command != "train":
            raise ValueError("Experiment does not exist in comet.ml!")
        logger.warning("Creating new experiment.")
        new_api_exp = api._create_experiment(
            workspace=comet_conf.workspace, project_name=project_name
        )
        exp_key = new_api_exp.id
    elif len(qres) == 1:
        logger.warning("Found existing experiment.")
        exp_key = qres[0].id
    else:
        raise ValueError("More then one experiment with the given hash.")
    logger.info(f"Experiment ID {exp_key}")

    experiment = comet_ml.ExistingExperiment(
        previous_experiment=exp_key,
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


def setup_writer():
    return SummaryWriter(conf.path.tensorboard)


def setup_experiment(model_holder):
    experiment = get_experiment()

    # Format the hyperparameter for comet
    hyperparameters_keyval_list = dict(dict_to_kv(hyperparameters))
    experiment.log_parameters(hyperparameters_keyval_list)
    for tag in set(conf.tag.split("_")):
        experiment.add_tag(tag)

    experiment.set_model_graph(str(model_holder.model))

    for part_name in conf.models:
        codepath = str(conf.models[part_name]["name"]).replace(".", "/")
        experiment.log_code(file_name=f"fgsim/models/{codepath}.py")

    experiment.set_step(model_holder.state["grad_step"])
    experiment.set_epoch(model_holder.state["epoch"])
    return experiment
