import os
import shutil
from typing import List

import comet_ml
from omegaconf import OmegaConf

from fgsim.config import conf
from fgsim.monitoring.monitor import get_exps_with_hash

if os.path.isdir(conf.path.run_path):
    shutil.rmtree(conf.path.run_path)


comet_conf = OmegaConf.load("fgsim/comet.yaml")
api = comet_ml.API(comet_conf.api_key)

qres: List[comet_ml.APIExperiment] = get_exps_with_hash(conf.hash)

for exp in qres:
    exp.archive()
