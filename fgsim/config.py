import hashlib
import os
import random
from typing import List

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fgsim.utils.cli import args


# Add a custum resolver to OmegaConf allowing for divisions
# Give int back if you can:
def divide(numerator, denominator):
    if numerator // denominator == numerator / denominator:
        return numerator // denominator
    else:
        return numerator / denominator


OmegaConf.register_new_resolver("div", divide, replace=True)

# Load the default settings, overwrite them
# witht the tag-specific settings and then
# overwrite those with cli arguments.

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

# Select the CPU/GPU


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda:" + str(torch.cuda.device_count() - 1))
        return dev
    else:
        return torch.device("cpu")


device = get_device()

# Exclude the keys that do not affect the training
def removekeys(omconf: DictConfig, excluded_keys: List[str]) -> DictConfig:
    filtered_omconf = OmegaConf.masked_copy(
        omconf,
        [k for k in omconf.keys() if k not in excluded_keys],
    )
    OmegaConf.resolve(filtered_omconf)
    return filtered_omconf


hyperparameters = removekeys(
    conf,
    ["command", "debug", "loglevel", "path"],
)
exclude_keys = ["preprocess_training", "debug"] + [
    x for x in conf["loader"] if "num_workers" in x
]
hyperparameters["loader"] = removekeys(hyperparameters["loader"], exclude_keys)


def gethash(omconf: DictConfig) -> str:
    omhash = str(hashlib.sha1(str(omconf).encode()).hexdigest()[:7])
    return omhash


conf["hash"] = gethash(conf)

# Conmpute a loader_hash
# this hash will be part of where the preprocessed
# dataset is safed to ensure the parameters dont change
# Exclude the keys that do not affect the training
conf["loader_hash"] = gethash(hyperparameters["loader"])

os.makedirs(conf.path.run_path, exist_ok=True)

if conf.command == "train":
    OmegaConf.save(hyperparameters, conf.path.train_config)


# Infer the parameters here
OmegaConf.resolve(conf)
OmegaConf.save(conf, conf.path.full_config)

torch.manual_seed(conf.seed)
np.random.seed(conf.seed)
random.seed(conf.seed)
