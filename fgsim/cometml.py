from comet_ml import ExistingExperiment, Experiment
from omegaconf import OmegaConf

from .config import hyperparameters

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
        experiment = ExistingExperiment(previous_experiment=exp_key, **comet_conf)
    else:
        experiment = Experiment(**comet_conf)
        exp_key = experiment.get_key()
        # Format the hyperparameter for comet

        hyperparametersD = dict(dict_to_kv(hyperparameters))

        experiment.log_parameters(hyperparametersD)
    return experiment
