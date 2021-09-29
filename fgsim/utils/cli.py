"""Console script for fgsim."""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--tag", default="default", required=False)
parser.add_argument(
    "--debug",
    dest="debug",
    help="Dump the old model",
    action="store_true",
    default=False,
    required=False,
)
subparsers = parser.add_subparsers(help="Available Commands", dest="command")

train_parser = subparsers.add_parser("train")
predict_parser = subparsers.add_parser("predict")
preprocess_parser = subparsers.add_parser("preprocess")
dump_parser = subparsers.add_parser("dump")
loadfile_parser = subparsers.add_parser("loadfile")
loadfile_parser.add_argument(
    "file_to_load",
    help="python file to load",
)

args = parser.parse_args()

if __name__ == "__main__":
    import sys

    for tag in args.tag.split(","):
        command = ""
        for e in sys.argv[1:]:
            if e is not args.tag:
                command += e + " "
            else:
                command += tag + " "
        print(args.command + " " + tag + " " + command)
