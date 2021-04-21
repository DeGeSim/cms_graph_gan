import os

from omegaconf import OmegaConf

from .cli import args


def get_device():
    import torch
    if torch.cuda.is_available():
        dev= torch.device("cuda")
        torch.cuda.set_device(3)
    else:
        dev = torch.device("cpu")
    return dev


device = get_device()

fn = f"wd/{args.tag}/config.yaml"
if not os.path.isfile(fn):
    fn = "fgsim/default.yaml"

with open(fn, "r") as fp:
    fileconf = OmegaConf.load(fp)

conf = OmegaConf.merge(vars(args), fileconf)
