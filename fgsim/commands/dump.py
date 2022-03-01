import os
import shutil
from typing import List

import comet_ml

from fgsim.config import conf
from fgsim.monitoring.monitor import get_exps_with_hash


def dump_procedure():
    if os.path.isdir(conf.path.run_path):
        shutil.rmtree(conf.path.run_path)

    qres: List[comet_ml.APIExperiment] = get_exps_with_hash(conf.hash)

    for exp in qres:
        exp.archive()
