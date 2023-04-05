import os
import shutil
from glob import glob

import wandb

from fgsim.monitoring.monitor import exp_orga_wandb
from fgsim.utils.cli import get_args


def dump_procedure():
    args = get_args()

    # wandb
    from fgsim.config import conf

    api = wandb.Api()
    run = api.run(f"{conf.project_name}/{exp_orga_wandb[conf.hash]}")
    run.delete()
    del exp_orga_wandb[args.hash]

    # local
    paths = glob(f"wd/*/{args.hash}")
    if len(paths) == 1:
        assert os.path.isdir(paths[0])
        shutil.rmtree(paths[0])
    elif len(paths) == 0:
        print("No directory found!")
    else:
        print("No directory found!")
        raise Exception
