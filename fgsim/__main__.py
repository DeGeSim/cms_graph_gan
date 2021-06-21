"""Main module."""
import importlib
import sys

import pretty_errors
from omegaconf import OmegaConf

from .utils.logger import logger


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

    logger.info("Configuration:\n" + OmegaConf.to_yaml(conf))

    logger.info(f"Running command {conf['command']}")

    if conf["command"] == "train":
        from .ml.holder import model_holder

        model_holder.load_checkpoint()
        from .ml.training import training_procedure

        training_procedure(model_holder)

    if conf["command"] == "predict":
        from .ml.holder import model_holder

        model_holder.load_best_model()
        from .ml.predict import prediction_procedure

        prediction_procedure(model_holder)

    if conf["command"] == "loadfile":
        fn = str(conf.file_to_load)
        import re

        fn = re.sub(".*fgsim/(.*?).py", ".\\1", fn)
        fn = re.sub("/", ".", fn)
        importlib.import_module(fn, "fgsim")


if __name__ == "__main__":
    main()
