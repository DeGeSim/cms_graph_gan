import os
from pathlib import Path
from shutil import copytree

from omegaconf import OmegaConf

from fgsim.config import conf


def setup_procedure() -> str:
    srcpath = Path(conf.path.run_path) / "fgsim"
    # If the experiment has been setup, exit directly
    if Path(conf.path.run_path).is_dir():
        return f"{conf.hash},{srcpath}"

    os.makedirs(conf.path.run_path, exist_ok=True)

    OmegaConf.save(conf, conf.path.full_config)

    # Backup the python files
    copytree(
        "fgsim",
        srcpath,
        ignore=lambda d, files: [f for f in files if filter_paths(d, f)],
        dirs_exist_ok=True,
    )

    from fgsim.monitoring.monitor import setup_experiment

    setup_experiment()
    return f"{conf.hash},{srcpath}"


def filter_paths(d, f):
    if f in ["old", "__pycache__"] or f.startswith("."):
        return True
    if (Path(d) / Path(f)).is_dir():
        return False
    if f.endswith(".py") or f.endswith(".yaml"):
        return False
    return True
