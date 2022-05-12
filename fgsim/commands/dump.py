import os
import shutil
from typing import List

import comet_ml

from fgsim.monitoring.monitor import get_exps_with_hash
from fgsim.utils.cli import args


def dump_procedure():
    if os.path.isfile(f"wd/{args.tag}/config.yaml"):
        from fgsim.config import conf

        if os.path.isdir(conf.path.run_path):
            shutil.rmtree(conf.path.run_path)

    qres: List[comet_ml.APIExperiment] = get_exps_with_hash(args.hash)

    for exp in qres:
        exp.archive()
