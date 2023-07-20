"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from pathlib import Path

import h5py
import torch
from torch_geometric.data import Batch
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import loader_info
from fgsim.ml.eval import postprocess
from fgsim.ml.holder import Holder


def generate_procedure() -> None:
    holder: Holder = Holder(device)
    holder.select_best_model()

    loader = QueuedDataset(loader_info)
    loader.qfseq.queue_iterable(loader.eval_chunks)
    loader.qfseq.start()

    generated_batches = []
    for sim_batch in tqdm(loader.qfseq):
        res = holder.pass_batch_through_model(sim_batch, train_disc=True)
        gen_batch = res["gen_batch"]

        __recur_transpant_dict(gen_batch, sim_batch)
        __recur_transpant_dict(gen_batch._slice_dict, sim_batch._slice_dict)
        __recur_transpant_dict(gen_batch._inc_dict, sim_batch._inc_dict)
        gen_batch = postprocess(gen_batch)
        generated_batches.append(gen_batch.to("cpu"))
        break

    data_list = []
    for gen_batch in generated_batches:
        data_list += gen_batch.to_data_list()
    gen_ds = Batch.from_data_list(data_list)

    your_energies = gen_ds.y.T[0]
    your_showers = gen_ds.x

    with h5py.File(Path(conf.path.run_path) / "out.hdf5", "w") as dataset_file:
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


def __recur_transpant_dict(gsd, ssd):
    for k, v in ssd.items():
        if isinstance(v, dict):
            if k not in gsd:
                gsd[k] = {}
            __recur_transpant_dict(gsd[k], ssd[k])
        elif isinstance(v, torch.Tensor):
            if k not in gsd:
                gsd[k] = v.clone()
        else:
            raise NotImplementedError(f"No definded behaviour for {type(v)}")
