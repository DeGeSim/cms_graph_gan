"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from pathlib import Path

import h5py
import torch
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import loader_info
from fgsim.loaders.calochallange.voxelize import voxelize
from fgsim.ml.eval import postprocess
from fgsim.ml.holder import Holder


def generate_procedure() -> None:
    holder: Holder = Holder(device)
    # holder.select_best_model()
    dspath = Path(conf.path.run_path) / "out.hdf5"
    if not dspath.exists():
        __write_dataset(holder, dspath)
    else:
        with h5py.File(dspath, "r") as ds:
            assert ds.attrs["hash"] == conf.hash
            if ds.attrs["grad_step"] != holder.state["grad_step"]:
                __write_dataset(holder, dspath)

    import os

    os.chdir("/home/mscham/homepage/code/")
    pathstr = f"{conf.tag}/{conf.hash}"
    for classifer in "cls-low", "cls-high", "cls-low-normed":
        command = (
            f"python evaluate.py -i {dspath} -m {classifer} -r"
            " ~/fgsim/data/calochallange2/dataset_2_2.hdf5 -d 2 --output_dir"
            f" ~/fgsim/wd/{pathstr}/cc_eval/"
        )
        os.system(command)

    exit(0)


def __write_dataset(holder, dspath):
    loader = QueuedDataset(loader_info)

    x_l = []
    E_l = []

    for sim_batch in tqdm(loader.eval_batches):
        batch_size = conf.loader.batch_size
        cond_gen_features = conf.loader.cond_gen_features

        if sum(cond_gen_features) > 0:
            cond = sim_batch.y[..., cond_gen_features].clone()
        else:
            cond = torch.empty((batch_size, 0)).float().to(device)
        gen_batch = holder.generate(cond, sim_batch.n_pointsv)

        gen_batch.y = sim_batch.y.clone()
        gen_batch = postprocess(gen_batch)

        # __recur_transpant_dict(gen_batch, sim_batch)
        # __recur_transpant_dict(gen_batch._slice_dict, sim_batch._slice_dict)
        # __recur_transpant_dict(gen_batch._inc_dict, sim_batch._inc_dict)
        x_l.append(voxelize(gen_batch).cpu())
        E_l.append(gen_batch.y.T[0].clone().cpu())

    your_energies = torch.hstack(E_l)
    your_showers = torch.vstack(x_l)

    with h5py.File(dspath, "w") as ds:
        ds.create_dataset(
            "incident_energies",
            data=your_energies.reshape(len(your_energies), -1),
            compression="gzip",
        )
        ds.create_dataset(
            "showers",
            data=your_showers.reshape(len(your_showers), -1),
            compression="gzip",
        )
        ds.attrs["hash"] = conf.hash
        ds.attrs["grad_step"] = holder.state["grad_step"]


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
