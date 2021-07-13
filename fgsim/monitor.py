import comet_ml
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .config import conf, hyperparameters

comet_conf = OmegaConf.load("fgsim/comet.yaml")


def dict_to_kv(o, keystr=""):
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


def get_experiment(exp_key):
    if exp_key is not None:
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
    else:
        experiment = comet_ml.Experiment(**comet_conf)
        exp_key = experiment.get_key()

        # Format the hyperparameter for comet

    experiment.set_name(hyperparameters["hash"])
    hyperparametersD = dict(dict_to_kv(hyperparameters))
    experiment.log_parameters(hyperparametersD)
    return experiment


def setup_writer():
    return SummaryWriter(conf.path.tensorboard)


def setup_experiment(model_holder):
    # Create an experiment with your api key
    if hasattr(model_holder.state, "comet_experiment_key"):
        experiment = get_experiment(model_holder.state.comet_experiment_key)
    else:
        experiment = comet_ml.Experiment(**comet_conf)
        model_holder.state.comet_experiment_key = experiment.get_key()

    experiment.set_model_graph(str(model_holder.model))
    experiment.log_code(file_name=f"fgsim/models/{conf.model.name}.py")

    return experiment
