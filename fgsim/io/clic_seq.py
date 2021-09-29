"""Here steps for reading the h5 files and processing the calorimeter images are definded.
`process_seq` is the function that should be passed the qfseq."""

import math
import os
from pathlib import Path

import h5py as h5
import numpy as np
import torch
import yaml
from torch.multiprocessing import Queue

from fgsim.config import conf

from . import qf

# Load files
ds_path = Path(conf.path.dataset)
assert ds_path.is_dir()
files = [str(e) for e in sorted(ds_path.glob("**/*.h5"))]
if len(files) < 1:
    raise RuntimeError("No hdf5 datasets found")


# load lengths
if not os.path.isfile(conf.path.ds_lenghts):
    len_dict = {}
    for fn in files:
        with h5.File(fn) as h5_file:
            len_dict[fn] = len(h5_file[conf.yvar])
    with open(conf.path.ds_lenghts, "w") as f:
        yaml.dump(len_dict, f, Dumper=yaml.SafeDumper)
else:
    with open(conf.path.ds_lenghts, "r") as f:
        len_dict = yaml.load(f, Loader=yaml.SafeLoader)

# reading from the filesystem
def read_chunk(chunks):
    data_dict = {k: [] for k in conf.loader.keylist}
    for chunk in chunks:
        file_path, start, end = chunk
        with h5.File(file_path) as h5_file:
            for k in conf.loader.keylist:
                data_dict[k].append(h5_file[k][start:end])
    for k in conf.loader.keylist:
        if len(data_dict[k][0].shape) == 1:
            data_dict[k] = np.hstack(data_dict[k])
        else:
            data_dict[k] = np.vstack(data_dict[k])

    # split up the events and pass them as a dict
    output = [
        {k: data_dict[k][ientry] for k in conf.loader.keylist}
        for ientry in range(conf.loader.chunksize)
    ]
    return output


def stack_batch(list_of_images):
    batch = {
        key: torch.stack(
            [torch.tensor(img[key], dtype=torch.float32) for img in list_of_images]
        )
        for key in conf.loader.keylist
    }
    return batch


def preprocessing(batch):
    windowSizeECAL = 25
    windowSizeHCAL = 11
    inputScaleSumE = 0.01
    inputScaleEta = 10.0

    # ECAL slice and energy sum
    ECAL = batch["ECAL"]
    lowerBound = math.ceil(ECAL.shape[1] / 2) - int(math.ceil(windowSizeECAL / 2))
    upperBound = lowerBound + windowSizeECAL
    ECAL = ECAL[:, lowerBound:upperBound, lowerBound:upperBound]
    ECAL = ECAL.contiguous().view(-1, 1, windowSizeECAL, windowSizeECAL, 25)

    ECAL_sum = ECAL.sum(2).sum(2).sum(2) * inputScaleSumE

    batch["ECAL"] = ECAL
    batch["ECAL_sum"] = ECAL_sum

    # HCAL slice to get energy sum
    HCAL = batch["HCAL"]
    lowerBound = math.ceil(HCAL.shape[1] / 2) - int(math.ceil(windowSizeHCAL / 2))
    upperBound = lowerBound + windowSizeHCAL
    HCAL = HCAL[:, lowerBound:upperBound, lowerBound:upperBound]
    HCAL = HCAL.contiguous().view(-1, 1, windowSizeHCAL, windowSizeHCAL, 60)

    HCAL_sum = HCAL.sum(2).sum(2).sum(2) * inputScaleSumE

    del batch["HCAL"]
    batch["HCAL_sum"] = HCAL_sum

    # reco angles
    batch["recoEta"] = batch["recoEta"].view(-1, 1) * inputScaleEta
    batch["recoPhi"] = batch["recoPhi"].view(-1, 1) * inputScaleEta

    return batch


def magic_do_nothing(elem):
    return elem


# Collect the steps
def process_seq():
    return (
        qf.ProcessStep(read_chunk, 4, name="read_chunk"),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(stack_batch, 1, name="stack_batch"),
        qf.ProcessStep(preprocessing, 4, name="preprocessing"),
        # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    )
