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

fn = f"wd/{args.tag}/config.yaml"
if not args.tag:
    fn = "fgsim/default.yaml"
if not os.path.isfile(fn):
    raise FileNotFoundError

with open(fn, "r") as fp:
    fileconf = OmegaConf.load(fp)

conf = OmegaConf.merge(vars(args), fileconf)

torch.manual_seed(conf.seed)
np.random.seed(conf.seed)
random.seed(conf.seed)
