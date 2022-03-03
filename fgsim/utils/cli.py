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
subparsers = parser.add_subparsers(help="Available Commands", dest="command")

setup_parser = subparsers.add_parser("setup")
train_parser = subparsers.add_parser("train")
test_parser = subparsers.add_parser("test")
preprocess_parser = subparsers.add_parser("preprocess")
dump_parser = subparsers.add_parser("dump")
loadfile_parser = subparsers.add_parser("loadfile")
loadfile_parser.add_argument(
    "file_to_load",
    help="python file to load",
)
# No args if run within pytest
if "pytest" in sys.modules:
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
