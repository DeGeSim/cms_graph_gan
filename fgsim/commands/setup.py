import os
from pathlib import Path
from shutil import copytree

from omegaconf import OmegaConf

from fgsim.config import conf, hyperparameters
from fgsim.monitoring.monitor import setup_experiment


def gethash_procedure() -> None:
    print(conf.hash)


def setup_procedure() -> str:
    rpath = Path(conf.path.run_path)

    # If the experiment has been setup, exit directly
    if rpath.is_dir():
        return conf.hash

    os.makedirs(conf.path.run_path, exist_ok=True)

    OmegaConf.save(conf, rpath / "conf.yaml")
    OmegaConf.save(hyperparameters, rpath / "hyperparameters.yaml")

    # Backup the python files
    copytree(
        "fgsim",
        rpath / "fgsim",
        ignore=lambda d, files: [f for f in files if filter_paths(d, f)],
        dirs_exist_ok=True,
    )
    copytree(
        "fgsim",
        rpath / "fgbackup",
        ignore=lambda d, files: [f for f in files if filter_paths(d, f)],
        dirs_exist_ok=True,
    )

    setup_experiment()
    return conf.hash


def filter_paths(d, f):
    if f in ["old", "__pycache__"] or f.startswith("."):
        return True
    if (Path(d) / Path(f)).is_dir():
        return False
    if f.endswith(".py") or f.endswith(".yaml"):
        return False
    return True
