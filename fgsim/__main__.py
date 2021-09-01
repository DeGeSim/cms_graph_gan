"""Main module."""
import importlib
import os
import sys

import comet_ml
import pretty_errors
from omegaconf import OmegaConf

from .utils.logger import logger

# Add the project to the path, -> `import fgsim.x`
sys.path.append(os.path.dirname(os.path.realpath(".")))


def main():
    # always reload the local modules
    # so that
    # `ipython >>> %run -m fgsim train`
    # works
    local_package_name = "fgsim"
    local_modules = {e for e in sys.modules if e.startswith(local_package_name)}
    do_not_reload = {
        # Never remove the upper packages
        "fgsim",
        # "fgsim.geo",
        # "fgsim.model",
        # Always reload cli and config
        # "fgsim.cli",
        # "fgsim.config",
        # utils dont change frequently
        "fgsim.utils",
        "fgsim.plot",
        # The rest
        "fgsim.geo.mapper",
        "fgsim.train.train",
        "fgsim.train.model",
        "fgsim.data_loader",
        "fgsim.train.holder",
        # Currently working on:
        # "fgsim.data_dumper",
        # "fgsim.geo.mapback",
        # "fgsim.train.generate",
    }
    for modulename in local_modules - do_not_reload:
        logger.info(f"Unloading {modulename}")
        del sys.modules[modulename]
    logger.info("Unloading complete")
    from .config import conf

    logger.info("Configuration:\n" + OmegaConf.to_yaml(conf, resolve=True))

    logger.info(f"Running command {conf['command']}")

    if conf["command"] == "train":
        from .ml.training import training_procedure

        training_procedure()

    if conf["command"] == "predict":
        from .ml.predict import prediction_procedure

        prediction_procedure()

    if conf["command"] == "profile":
        from .ml.profile import profile_procedure

        profile_procedure()

    if conf["command"] == "loadfile":
        fn = str(conf.file_to_load)
        import re

        fn = re.sub(".*fgsim/(.*?).py", ".\\1", fn)
        fn = re.sub("/", ".", fn)
        importlib.import_module(fn, "fgsim")

    if conf["command"] == "dump":
        from .utils import dump_training


if __name__ == "__main__":
    main()
