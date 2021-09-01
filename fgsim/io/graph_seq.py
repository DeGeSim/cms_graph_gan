"""
Here steps for reading the h5 files and processing the calorimeter \
images to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""
import h5py as h5
import numpy as np
import torch_geometric
from torch.multiprocessing import Queue

from fgsim.config import conf
from fgsim.geo.batch_stack import split_layer_subgraphs
from fgsim.geo.transform import transform

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


def geo_batch(list_of_graphs):
    batch = torch_geometric.data.Batch().from_data_list(list_of_graphs)
    return batch


def magic_do_nothing(elem):
    return elem


# Collect the steps
def process_seq():
    return (
        qf.ProcessStep(read_chunk, 2, name="read_chunk"),
        Queue(1),
        # In the input is now [(x,y), ... (x [300 * 51 * 51 * 25], y [300,1] ), (x,y)]
        # For these elements to be processed by each of the workers in the following
        # transformthey need to be (x [51 * 51 * 25], y [1] ):
        qf.PoolStep(
            transform, nworkers=conf.loader.num_workers_transform, name="transform"
        ),
        Queue(1),
        qf.RepackStep(conf.loader.batch_size),
        qf.ProcessStep(geo_batch, 1, name="geo_batch"),
        qf.ProcessStep(
            split_layer_subgraphs,
            conf.loader.num_workers_stack,
            name="split_layer_subgraphs",
        ),
        # Needed for outputs to stay in order.
        qf.ProcessStep(
            magic_do_nothing,
            1,
            name="magic_do_nothing",
        ),
        Queue(conf.loader.prefetch_batches),
    )
