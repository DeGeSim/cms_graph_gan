# %%
import os
import shutil
from glob import glob
from pathlib import Path

from omegaconf import OmegaConf

import wandb
from fgsim.monitoring.experiment_organizer import ExperimentOrganizer
from fgsim.utils.oc_utils import dict_to_kv

os.chdir(Path("~/fgsim").expanduser())
project = "calochallange"
exp_orga_wandb = ExperimentOrganizer("wandb")
api = wandb.Api()


def del_wandb_dirs(exphash):
    folder = get_folder(exphash)
    wandb_dir = folder / "wandb"
    if wandb_dir.exists():
        shutil.rmtree(wandb_dir)


def get_confs(exphash):
    folder = get_folder(exphash)
    hyperparameters = OmegaConf.load(folder / "hyperparameters.yaml")
    conf = OmegaConf.load(folder / "conf.yaml")
    return conf, hyperparameters


def get_folder(exphash):
    globstr = f"wd/*/{exphash}/"
    folder = Path(glob(globstr)[0])
    return folder


def del_runs(exphash):
    for s in [exphash, f"{exphash}_train", f"{exphash}_test"]:
        if s in exp_orga_wandb:
            try:
                run = api.from_path(f"{project}/runs/{exp_orga_wandb[s]}")
                run.delete()
                del exp_orga_wandb[s]
            except wandb.errors.CommError:
                pass
    print(exphash)


def reinit_runs(exphash):
    conf, hyperparameters = get_confs(exphash)

    hyperparameters_keyval_list = dict(dict_to_kv(hyperparameters))
    hyperparameters_keyval_list["hash"] = conf["hash"]
    hyperparameters_keyval_list["loader_hash"] = conf["loader_hash"]
    tags_list = list(set(conf.tag.split("_")))

    # wandb
    run_train = wandb.init(
        project=conf.project_name,
        name=f"{conf['hash']}_train",
        group=conf["hash"],
        tags=tags_list,
        config=hyperparameters_keyval_list,
        dir=conf.path.run_path,
        job_type="train",
        resume=False,
        reinit=True,
        settings={
            "quiet": True,
            "disable_job_creation": True,
            "code_dir": f"./{conf.path.run_path}/fgsim/models",
        },
    )
    codepath = Path(conf.path.run_path) / "fgsim/models"
    assert codepath.is_dir()
    run_train.log_code(codepath.absolute())
    exp_orga_wandb[conf["hash"]] = run_train.id

    run_test = wandb.init(
        project=conf.project_name,
        name=f"{conf['hash']}_test",
        group=conf["hash"],
        tags=tags_list,
        config=hyperparameters_keyval_list,
        dir=conf.path.run_path,
        job_type="test",
        reinit=True,
        resume=False,
        settings={"quiet": True, "disable_job_creation": True},
    )
    exp_orga_wandb[f"{conf['hash']}_test"] = run_test.id


for h in ["040f723", "f7fbe2b", "94ddd08", "b0e90d4", "c4d5763"]:
    print(h)
    del_runs(h)
    del_wandb_dirs(h)
    reinit_runs(h)
