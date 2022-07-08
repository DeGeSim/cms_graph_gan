import os
import shutil
from glob import glob

from fgsim.monitoring.monitor import api_experiment_from_hash, exp_orga
from fgsim.utils.cli import args


def dump_procedure():
    api_experiment_from_hash(args.hash).archive()
    paths = glob(f"wd/*/{args.hash}")
    del exp_orga.d[args.hash]
    exp_orga.save()
    if len(paths) == 1:
        assert os.path.isdir(paths[0])
        shutil.rmtree(paths[0])
    elif len(paths) == 0:
        print("No directory found!")
    else:
        print("No directory found!")
        raise Exception
