"""Main module."""
import importlib
import os
import sys
from pathlib import Path

# Add the project to the path, -> `import fgsim.x`
sys.path.append(os.path.dirname(os.path.realpath(".")))

# from typeguard.importhook import install_import_hook

# install_import_hook("fgsim")


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

    from fgsim.utils.cli import get_args

    args = get_args()

    if args.command not in ["setup", "dump", "overwrite"]:
        import comet_ml  # noqa
        import pretty_errors  # noqa

    import fgsim.config

    (
        fgsim.config.conf,
        fgsim.config.hyperparameters,
    ) = fgsim.config.parse_arg_conf()

    # If it is called by the hash, manipulate then
    overwrite_path_bool = (
        args.command not in ["gethash", "setup", "dump", "overwrite"]
        and not args.debug
        and not args.ray
    )
    if overwrite_path_bool:
        old_path, new_fgsim_path = overwrite_path()

    if args.command not in ["gethash", "setup", "dump", "overwrite"]:
        from fgsim.config import conf
        from fgsim.monitoring.logger import init_logger, logger

        init_logger()
        if overwrite_path_bool:
            logger.warning(f"Replaced path {old_path} with {new_fgsim_path}.")

        logger.info(
            f"tag: {conf.tag} hash: {conf.hash} loader_hash: {conf.loader_hash}"
        )
        logger.info(f"Running command {args.command}")

    if args.command == "gethash":
        if args.hash is not None:
            raise Exception
        from fgsim.commands.setup import gethash_procedure

        gethash_procedure()

    elif args.command == "setup":
        if args.hash is not None:
            raise Exception
        from fgsim.commands.setup import setup_procedure

        print(setup_procedure())
    elif args.command == "dump":
        from fgsim.commands.dump import dump_procedure

        dump_procedure()

    elif args.command == "overwrite":
        from fgsim.commands.overwrite import overwrite_procedure

        overwrite_procedure()

    elif args.command == "preprocess":
        from fgsim.commands.preprocess import preprocess_procedure

        preprocess_procedure()

    elif args.command == "train":
        from fgsim.commands.training import training_procedure

        training_procedure()

    elif args.command == "test":
        from fgsim.commands.testing import test_procedure

        test_procedure()

    elif args.command == "loadfile":
        file_name = str(conf.file_to_load)
        import re

        file_name = re.sub(".*fgsim/(.*?).py", ".\\1", file_name)
        file_name = re.sub("/", ".", file_name)
        importlib.import_module(file_name, "fgsim")

    else:
        raise Exception


def overwrite_path():
    from fgsim.config import conf

    new_fgsim_path = str((Path(conf.path.run_path)).absolute())
    if not (Path(conf.path.run_path) / "fgsim").is_dir():
        raise Exception("setup has not been executed")
    del conf
    del sys.modules["fgsim"]
    #
    pathlist = [e for e in sys.path if e.endswith("fgsim")]
    # make sure that this is unique
    if len({e for e in pathlist}) == 0:
        old_path = ""
    elif len({e for e in pathlist}) == 1:
        # remove the old path
        old_path = pathlist[0]
        for path in pathlist:
            sys.path.remove(path)
    elif len({e for e in pathlist}) > 1:
        raise Exception
    sys.path.insert(0, new_fgsim_path)

    return old_path, new_fgsim_path


if __name__ == "__main__":
    main()
