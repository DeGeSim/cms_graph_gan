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
train_parser.add_argument(
    "-d",
    "--dump_model",
    help="Dump the old model",
    action="store_true",
    default=False,
    required=False,
)
predict_parser = subparsers.add_parser("predict")
loadfile_parser = subparsers.add_parser("loadfile")
loadfile_parser.add_argument(
    "file_to_load",
    help="python file to load",
)

args = parser.parse_args()
