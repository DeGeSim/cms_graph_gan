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
commands = [
    "setup",
    "gethash",
    "train",
    "test",
    "preprocess",
    "dump",
    "overwrite",
    "loadfile",
]
subparsers = parser.add_subparsers(help="Available Commands", dest="command")

commandparsers = {command: subparsers.add_parser(command) for command in commands}

commandparsers["loadfile"].add_argument(
    "file_to_load",
    help="python file to load",
)
# No args if run within pytest
if any([x in sys.modules for x in ["IPython", "pytest", "ray"]]):
    argv = [
        # "/home/mscham/fgsim/fgsim/__main__.py",
        "--tag",
        "default",
        "train",
    ]
    args = parser.parse_args(argv)

else:
    args = parser.parse_args()

if __name__ == "__main__":
    import sys

    if args.hash is None:
        for tag in args.tag.split(","):
            command = ""
            for e in sys.argv[1:]:
                if e is not args.tag:
                    command += e + " "
                else:
                    command += tag + " "
            print(args.command + " " + tag + " " + command)
    else:
        for tag in args.hash.split(","):
            command = ""
            for e in sys.argv[1:]:
                if e is not args.hash:
                    command += e + " "
                else:
                    command += tag + " "
            print(args.command + " " + tag + " " + command)
