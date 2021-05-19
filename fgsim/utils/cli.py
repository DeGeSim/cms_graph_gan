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

train_parser = subparsers.add_parser("generate")
geo_parser = subparsers.add_parser("geo")
trytest_parser = subparsers.add_parser("trytest")
write_sparse_ds_parser = subparsers.add_parser("write_sparse_ds")


args = parser.parse_args()
