import os
import shutil
from glob import glob

import wandb

from fgsim.monitoring.monitor import (
    comet_api,
    exp_orga_comet,
    exp_orga_wandb,
    search_experiement_by_name,
)
from fgsim.utils.cli import get_args


def dump_procedure():
    args = get_args()

    # comet_ml
    if args.hash in exp_orga_comet.keys():
        # try to archive the experiment:
        try:
            comet_api.get_experiment_by_key(exp_orga_comet[args.hash]).archive()
        except KeyError:
            pass
        del exp_orga_comet[args.hash]
    else:
        for exp in search_experiement_by_name(args.hash):
            exp.archive()

    # wandb
    from fgsim.config import conf

    api = wandb.Api()
    run = api.run(f"{conf.comet_project_name}/{exp_orga_wandb[conf.hash]}")
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
