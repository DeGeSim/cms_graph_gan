"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys

import caloutils

from fgsim.config import conf

if "dataset_2" in conf.loader.dataset_glob:
    caloutils.init_calorimeter("cc_ds2")
elif "dataset_3" in conf.loader.dataset_glob:
    caloutils.init_calorimeter("cc_ds3")
else:
    raise Exception("No such dataset")

from .dataset import Dataset
from .postprocess import postprocess
from .scaler import scaler
