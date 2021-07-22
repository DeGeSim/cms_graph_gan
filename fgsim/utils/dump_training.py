import os
import shutil

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
