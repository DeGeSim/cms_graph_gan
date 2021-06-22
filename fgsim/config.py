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


if args.tag:
    fn = f"wd/{args.tag}/config.yaml"
    if not os.path.isfile(fn):
        raise FileNotFoundError
    with open(fn, "r") as fp:
        tagconf = OmegaConf.load(fp)
else:
    tagconf = OmegaConf.create({})

conf = OmegaConf.merge(defaultconf, tagconf, vars(args))

if conf.command == "train":
    OmegaConf.save(conf, conf.path.train_config)

torch.manual_seed(conf.seed)
np.random.seed(conf.seed)
random.seed(conf.seed)
