"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""

from pathlib import Path
from subprocess import PIPE, CalledProcessError, Popen

import h5py
import torch
from caloutils.processing import pc_to_voxel
from tqdm import tqdm

from fgsim.config import conf, device
from fgsim.io.queued_dataset import QueuedDataset
from fgsim.io.sel_loader import loader_info
from fgsim.ml.eval import postprocess
from fgsim.ml.holder import Holder
from fgsim.monitoring import logger

conf.command = "test"


def generate_procedure() -> None:
    holder: Holder = Holder(device)
    for best_or_last in ["last", "best"]:
        if best_or_last == "best":
            holder.checkpoint_manager.select_best_model()

        dspath = Path(conf.path.run_path).absolute() / f"out_{best_or_last}.hdf5"
        if not dspath.exists():
            __write_dataset(holder, dspath)
        else:
            with h5py.File(dspath, "r") as ds:
                if "hash" not in ds.attrs or "grad_step" not in ds.attrs:
                    __write_dataset(holder, dspath)
                assert ds.attrs["hash"] == conf.hash
                if ds.attrs["grad_step"] != holder.state["grad_step"]:
                    __write_dataset(holder, dspath)

        ## compute the aucs
        resd = __run_classifiers(dspath)

        holder.train_log.log_metrics(
            resd,
            prefix="/".join(["test", best_or_last]),
            step=holder.state["grad_step"],
            epoch=holder.state["epoch"],
        )
        holder.train_log.wandb_run.summary.update(
            {"/".join(["m", "test", best_or_last, k]): v for k, v in resd.items()}
        )
        logger.info(resd)
        holder.train_log.flush()
    exit(0)


def __run_classifiers(dspath):
    resd = {}

    rpath = Path(conf.path.run_path).absolute()
    outdir = rpath / "cc_eval/"
    test_path = "/home/mscham/fgsim/data/calochallange2/dataset_2_2.hdf5"
    for classifer in "cls-high", "cls-low", "cls-low-normed":
        logger.info(f"Running classifier {classifer}")
        cmd = (
            f"/dev/shm/mscham/fgsim/bin/python evaluate.py -i {dspath} -m"
            f" {classifer} -r {test_path} -d 2"
            f" --output_dir {outdir}"
        )

        lines = []
        with Popen(
            cmd.split(" "),
            stdout=PIPE,
            bufsize=1,
            universal_newlines=True,
            cwd="/home/mscham/homepage/code/",
        ) as p:
            for line in p.stdout:
                lines.append(line)
                # print(line, end="")

        if p.returncode != 0:
            raise CalledProcessError(p.returncode, p.args)

        # aucidx = lines.index("Final result of classifier test (AUC / JSD):\n") + 1
        auc = float(lines[-1].split("/")[0].rstrip())
        resd[classifer] = auc
        logger.info(f"Classifier {classifer} AUC {auc}")
    return resd


def __write_dataset(holder, dspath):
    loader = QueuedDataset(loader_info)

    x_l = []
    E_l = []

    for sim_batch in tqdm(loader.eval_batches, miniters=100, mininterval=5.0):
        sim_batch = sim_batch.clone().to(device)
        batch_size = conf.loader.batch_size
        cond_gen_features = conf.loader.cond_gen_features

        if sum(cond_gen_features) > 0:
            cond = sim_batch.y[..., cond_gen_features].clone()
        else:
            cond = torch.empty((batch_size, 0)).float().to(device)
        gen_batch = holder.generate(cond, sim_batch.n_pointsv)

        gen_batch.y = sim_batch.y.clone()
        gen_batch = postprocess(gen_batch, "gen")

        # __recur_transpant_dict(gen_batch, sim_batch)
        # __recur_transpant_dict(gen_batch._slice_dict, sim_batch._slice_dict)
        # __recur_transpant_dict(gen_batch._inc_dict, sim_batch._inc_dict)
        x_l.append(pc_to_voxel(gen_batch).cpu())
        E_l.append(gen_batch.y.T[0].clone().cpu())
        del sim_batch

    your_energies = torch.hstack(E_l)
    your_showers = torch.vstack(x_l)

    if dspath.exists():
        dspath.unlink()
    # dspath.touch()
    with h5py.File(dspath.absolute(), "w") as ds:
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
