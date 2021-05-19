import os

import numpy as np
import torch
from omegaconf import OmegaConf

from .utils.cli import args


def get_device():

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        torch.cuda.set_device(3)
    else:
        dev = torch.device("cpu")
    return dev


torch.manual_seed(0)
np.random.seed(0)

device = get_device()

fn = f"wd/{args.tag}/config.yaml"
if not os.path.isfile(fn):
    fn = "fgsim/default.yaml"

with open(fn, "r") as fp:
    fileconf = OmegaConf.load(fp)

conf = OmegaConf.merge(vars(args), fileconf)
