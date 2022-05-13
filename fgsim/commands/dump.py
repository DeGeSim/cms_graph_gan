import os
import shutil
from glob import glob
from typing import List

import comet_ml

from fgsim.monitoring.monitor import get_exps_with_hash
from fgsim.utils.cli import args


def dump_procedure():
    qres: List[comet_ml.APIExperiment] = get_exps_with_hash(args.hash)
    for exp in qres:
        exp.archive()
    paths = glob(f"wd/*/{args.hash}")
    if len(paths) == 1:
        assert os.path.isdir(paths[0])
        shutil.rmtree(paths[0])
    elif len(paths) == 0:
        print("No directory found!")
    else:
        print("No directory found!")
        raise Exception
