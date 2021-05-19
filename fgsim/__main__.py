"""Main module."""
import sys

import pretty_errors

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

    logger.info(f"Running command {conf['command']}")

    if conf["command"] == "train":
        from .model.training import training_procedure
        from .model.holder import model_holder

        training_procedure(model_holder)

    if conf["command"] == "generate":
        from .model.generate import generation_procedure
        from .model.holder import model_holder

        generation_procedure(model_holder)

    if conf["command"] == "trytest":
        from .torchdata_loader import dataset

        print(len(dataset))

    if conf["command"] == "write_sparse_ds":
        from .write_sparse_ds import write_sparse_ds

        write_sparse_ds()


if __name__ == "__main__":
    main()
