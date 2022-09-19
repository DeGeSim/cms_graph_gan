import os
import shutil
from glob import glob

from fgsim.monitoring.monitor import (
    api_experiment_from_hash,
    exp_orga,
    search_experiement_by_name,
)
from fgsim.utils.cli import get_args


def dump_procedure():
    args = get_args()
    if args.hash in exp_orga.d:
        # try to archive the experiment:
        try:
            api_experiment_from_hash(args.hash).archive()
        except KeyError:
            pass
        del exp_orga.d[args.hash]
    else:
        for exp in search_experiement_by_name(args.hash):
            exp.archive()
    exp_orga.save()

    paths = glob(f"wd/*/{args.hash}")
    if len(paths) == 1:
        assert os.path.isdir(paths[0])
        shutil.rmtree(paths[0])
    elif len(paths) == 0:
        print("No directory found!")
    else:
        print("No directory found!")
        raise Exception
