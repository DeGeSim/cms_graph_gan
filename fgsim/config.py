import hashlib
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from .utils.cli import args


def get_device():

    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    return dev


device = get_device()

with open("fgsim/default.yaml", "r") as fp:
    defaultconf = OmegaConf.load(fp)


if args.tag != "default":
    fn = f"wd/{args.tag}/config.yaml"
    if not os.path.isfile(fn):
        raise FileNotFoundError
    with open(fn, "r") as fp:
        tagconf = OmegaConf.load(fp)
else:
    tagconf = OmegaConf.create({})

conf = OmegaConf.merge(defaultconf, tagconf, vars(args))


# Exclude the keys that do not affect the training
hyperparameters = OmegaConf.masked_copy(
    conf,
    [
        k
        for k in conf.keys()
        if k not in ["command", "dump_model", "debug", "loglevel", "path"]
    ],
)

OmegaConf.resolve(hyperparameters)

# Compute the hash
conf_hash = hashlib.sha1(str(hyperparameters).encode()).hexdigest()[:7]
conf["hash"] = conf_hash
hyperparameters["hash"] = conf_hash

# Format the hyperparameter for comet
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


hyperparametersD = dict(dict_to_kv(hyperparameters))


os.makedirs(conf.path.run_path, exist_ok=True)


if conf.command == "train":
    OmegaConf.save(hyperparameters, conf.path.train_config)

# Infer the parameters here
OmegaConf.resolve(conf)


comet_conf = OmegaConf.load("fgsim/comet.yaml")


torch.manual_seed(conf.seed)
np.random.seed(conf.seed)
random.seed(conf.seed)
