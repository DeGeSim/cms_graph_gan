"""Main module."""
import importlib
import os
import sys
from pathlib import Path

import comet_ml  # noqa
import pretty_errors  # noqa

from fgsim.utils.cli import args

# Add the project to the path, -> `import fgsim.x`
sys.path.append(os.path.dirname(os.path.realpath(".")))

from typeguard.importhook import install_import_hook

install_import_hook("fgsim")


def main():
    # # always reload the local modules
    # # so that
    # # `ipython >>> %run -m fgsim train`
    # # works
    # local_package_name = "fgsim"
    # local_modules = {e for e in sys.modules if e.startswith(local_package_name)}
    # do_not_reload = {
    #     # Never remove the upper packages
    #     "fgsim",
    #     # "fgsim.geo",
    #     # "fgsim.model",
    #     # Always reload cli and config
    #     # "fgsim.cli",
    #     # "fgsim.config",
    #     # utils dont change frequently
    #     "fgsim.utils",
    #     "fgsim.plot",
    #     # The rest
    #     "fgsim.geo.mapper",
    #     "fgsim.train.train",
    #     "fgsim.train.model",
    #     "fgsim.data_loader",
    #     "fgsim.train.holder",
    #     # Currently working on:
    #     # "fgsim.data_dumper",
    #     # "fgsim.geo.mapback",
    #     # "fgsim.train.generate",
    # }
    # for modulename in local_modules - do_not_reload:
    #     logger.info(f"Unloading {modulename}")
    #     del sys.modules[modulename]
    # logger.info("Unloading complete")

    # If we are running from tags, run the setup
    if args.command == "train" and args.hash is None:
        from fgsim.commands.setup import setup_procedure

        setup_procedure()
    if args.command == "setup":
        from fgsim.commands.setup import setup_procedure

        print(setup_procedure())
        exit()

    # If it is called by the hash, manipulate then
    if args.hash is not None:
        from fgsim.config import conf

        new_fgsim_path = str((Path(conf.path.run_path)).absolute())
        del conf
        del sys.modules["fgsim"]
        #
        pathlist = [e for e in sys.path if e.endswith("fgsim")]
        # make sure that this is unique
        if len(pathlist) != 1:
            raise Exception
        # remove the old path
        old_path = pathlist[0]
        sys.path.remove("")
        sys.path.remove(pathlist[0])
        sys.path.append(new_fgsim_path)
        from fgsim.monitoring.logger import logger

        logger.warning(f"Replaced path {old_path} with {new_fgsim_path}.")

    from fgsim.config import conf
    from fgsim.monitoring.logger import logger

    logger.warning(
        f"Using hash {conf['hash']} and loader_hash {conf['loader_hash']}"
    )
    logger.info(f"Running command {args.command}")

    if args.command == "train":
        from fgsim.commands.training import training_procedure

        training_procedure()

    if args.command == "test":
        from fgsim.commands.testing import test_procedure

        test_procedure()

    if args.command == "preprocess":
        from fgsim.commands.preprocess import preprocess_procedure

        preprocess_procedure()

    if args.command == "loadfile":
        file_name = str(conf.file_to_load)
        import re

        file_name = re.sub(".*fgsim/(.*?).py", ".\\1", file_name)
        file_name = re.sub("/", ".", file_name)
        importlib.import_module(file_name, "fgsim")

    if args.command == "dump":
        from fgsim.commands.dump import dump_procedure

        dump_procedure()


if __name__ == "__main__":
    main()
