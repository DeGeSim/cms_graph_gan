"""Here steps for reading the h5 files and processing the calorimeter images are definded.
`process_seq` is the function that should be passed the qfseq."""

import math

import h5py as h5
import numpy as np
import torch
from torch.multiprocessing import Queue

from ..config import conf
from . import qf


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
