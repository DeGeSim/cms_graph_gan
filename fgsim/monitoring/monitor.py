import wandb
from fgsim.utils.oc_utils import dict_to_kv

from .experiment_organizer import ExperimentOrganizer

exp_orga_wandb = ExperimentOrganizer("wandb")


def setup_experiment() -> None:
    from fgsim.config import conf

    """Generates a new experiment."""
    if conf.hash in exp_orga_wandb.keys():
        if conf.ray:
            return
        raise Exception("Experiment exists")

    # Format the hyperparameter
    from fgsim.config import hyperparameters

    assert len(hyperparameters) > 0
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
    run_train.log_code(f"./{conf.path.run_path}/fgsim/models")
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
