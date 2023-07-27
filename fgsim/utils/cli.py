"""Console script for fgsim."""
import argparse
import sys

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()

group.add_argument(
    "-t",
    "--tag",
    default="default",
)
group.add_argument("--hash")

parser.add_argument(
    "--debug",
    dest="debug",
    action="store_true",
    default=False,
    required=False,
)
parser.add_argument(
    "--ray",
    dest="ray",
    action="store_true",
    default=False,
    required=False,
)
parser.add_argument(
    "--remote",
    dest="remote",
    action="store_true",
    default=False,
    required=False,
)
commands = [
    "setup",
    "gethash",
    "train",
    "test",
    "generate",
    "preprocess",
    "dump",
    "overwrite",
    "loadfile",
    "implant_checkpoint",
]
subparsers = parser.add_subparsers(help="Available Commands", dest="command")

commandparsers = {command: subparsers.add_parser(command) for command in commands}

commandparsers["loadfile"].add_argument(
    "file_to_load",
    help="python file to load",
)


def get_args():
    return parser.parse_args()
    # No args if run within pytest
    # for ipython / jupyter : "IPython" remove for scalene
    if any(
        [x in sys.modules for x in ["IPython", "jupyter_core", "pytest", "ray"]]
    ):
        argv = [
            # "/home/mscham/fgsim/fgsim/__main__.py",
            "--tag",
            "default",
            "train",
        ]
        args = parser.parse_args(argv)

    else:
        args = parser.parse_args()
    return args
