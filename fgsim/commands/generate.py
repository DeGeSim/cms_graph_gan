"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""


import h5py

from fgsim.config import device
from fgsim.ml.holder import Holder

from .testing import get_testing_datasets


def generate_procedure() -> None:
    holder: Holder = Holder(device)
    ds = get_testing_datasets(holder, "best").res_d["gen_batch"]
    your_energies = ds.y.T[0]
    your_showers = ds.x

    with h5py.File("out.hdf5", "w") as dataset_file:
        dataset_file.create_dataset(
            "incident_energies",
            data=your_energies.reshape(len(your_energies), -1),
            compression="gzip",
        )
        dataset_file.create_dataset(
            "showers",
            data=your_showers.reshape(len(your_showers), -1),
            compression="gzip",
        )

    exit(0)
