import hashlib
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from .utils.cli import args

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


def get_device():
    if torch.cuda.is_available():  # and not conf.profile:
        # if torch.cuda.device_count() > 1:
        #     dev = torch.device("cuda:" + str(torch.cuda.device_count() - 1))
        # else:
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")
    return dev


device = get_device()

# Exclude the keys that do not affect the training
hyperparameters = OmegaConf.masked_copy(
    conf,
    [k for k in conf.keys() if k not in ["command", "debug", "loglevel", "path"]],
)

OmegaConf.resolve(hyperparameters)

# Compute the hash
conf_hash = hashlib.sha1(str(hyperparameters).encode()).hexdigest()[:7]
conf["hash"] = conf_hash
hyperparameters["hash"] = conf_hash

os.makedirs(conf.path.run_path, exist_ok=True)

if conf.command == "train":
    OmegaConf.save(hyperparameters, conf.path.train_config)

# Infer the parameters here
OmegaConf.resolve(conf)


torch.manual_seed(conf.seed)
np.random.seed(conf.seed)
random.seed(conf.seed)
