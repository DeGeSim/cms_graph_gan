import os
import shutil

import comet_ml
from omegaconf import OmegaConf

from ..config import conf

filenames = [
    conf.path[key]
    for key in [
        "log",
        "train_config",
        "checkpoint",
        "checkpoint_old",
        "predict_csv",
    ]
]

for fn in filenames:
    if os.path.isfile(fn):
        os.remove(fn)

if os.path.isdir(conf.path.tensorboard):
    shutil.rmtree(conf.path.tensorboard)


comet_conf = OmegaConf.load("fgsim/comet.yaml")
api = comet_ml.API(comet_conf.api_key)

experiments = [
    exp
    for exp in api.get(
        workspace=comet_conf.workspace, project_name=comet_conf.project_name
    )
]
qres = [
    exp
    for exp in experiments
    if exp.get_parameters_summary("hash")["valueCurrent"] == conf.hash
]
api.delete_experiments([exp.id for exp in qres])
