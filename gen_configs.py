from collections import defaultdict
from dataclasses import dataclass
from os import mkdir
from typing import Callable, Dict, List

from genericpath import isdir
from omegaconf import DictConfig, OmegaConf


@dataclass
class ExperimentConfig:
    config: DictConfig
    tags: List[str]


@dataclass
class ExperimentOption:
    config_change: Dict[str, DictConfig]


base_config = ExperimentConfig(
    OmegaConf.create(
        {
            "models": {
                "gen": {
                    "optim": {"params": {}},
                    "losses_list": ["CEGenLoss", "cd"],
                },
                "disc": {
                    "optim": {"params": {}},
                    "losses_list": ["CEDiscLoss"],
                },
            },
        }
    ),
    ["jetnet", "childconf"],
)

exp_list: List[ExperimentConfig] = [base_config]

optionslist: List[Callable] = []


def add_option(option: Callable):
    optionslist.append(option)


@add_option
def option_gantype(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["CE"]
    res["W"]["models"]["gen"]["losses_list"] = ["WGenLoss", "cd"]
    res["W"]["models"]["disc"]["losses_list"] = [
        "WDiscLoss",
        "GradientPenalty",
    ]
    return res


# @add_option
# def option_optimizer(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["Adam"]["models"]["gen"]["optim"]["name"] = "Adam"
#     res["RMSprop"]["models"]["gen"]["optim"]["name"] = "RMSprop"
#     return res


# @add_option
# def option_weight_decay(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     if exp_config.config["models"]["gen"]["optim"]["name"] != "Adam":
#         return res
#     res["wdecay"]["models"]["gen"]["optim"]["params"]["weight_decay"] = 1.0e-4
#     res["wdecay"]["models"]["disc"]["optim"]["params"]["weight_decay"] = 1.0e-4
#     res["nodecay"]["models"]["gen"]["optim"]["params"]["weight_decay"] = 0
#     res["nodecay"]["models"]["disc"]["optim"]["params"]["weight_decay"] = 0
#     return res


# @add_option
# def option_disc_lr(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     # res["lrddef"]["models"]["disc"]["optim"]["params"]["lr"] = 2.0e-4
#     # res["lrdlow"]["models"]["disc"]["optim"]["params"]["lr"] = 2.0e-5
#     res["lrdvlow"]["models"]["disc"]["optim"]["params"]["lr"] = 2.0e-6
#     res["lrdvvlow"]["models"]["disc"]["optim"]["params"]["lr"] = 5.0e-7
#     return res


# @add_option
# def option_gen_lr(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     # res["lrgdef"]["models"]["gen"]["optim"]["params"]["lr"] = 2.0e-4
#     # res["lrglow"]["models"]["gen"]["optim"]["params"]["lr"] = 6.0e-5
#     res["lrgvlow"]["models"]["gen"]["optim"]["params"]["lr"] = 1.0e-5
#     res["lrgvvlow"]["models"]["gen"]["optim"]["params"]["lr"] = 1.0e-6
#     return res


# @add_option
# def option_dist(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["mmdpc"]["models"]["gen"]["losses_list"] = ["CEGenLoss", "mmdpc"]
#     res["dcd"]["models"]["gen"]["losses_list"] = ["CEGenLoss", "dcd"]
#     res["cd"]["models"]["gen"]["losses_list"] = ["CEGenLoss", "cd"]
#     return res


# @add_option
# def option_batches(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     distname = exp_config.config["models"]["gen"]["losses_list"][1]
#     res["bw"]["loss_options"] = {distname: {"batch_wise": True}}
#     res["single"]["loss_options"] = {distname: {"batch_wise": False}}
#     return res


# @add_option
# def option_norm(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     distname = exp_config.config["models"]["gen"]["losses_list"][1]
#     if distname not in ["dcd", "cd"]:
#         return res
#     res["l1"]["loss_options"][distname]["lpnorm"] = 1
#     res["l2"]["loss_options"][distname]["lpnorm"] = 2
#     return res


# @add_option
# def option_pow(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     distname = exp_config.config["models"]["gen"]["losses_list"][1]
#     if distname not in ["dcd", "cd"]:
#         return res
#     res["pow1"]["loss_options"][distname]["pow"] = 1
#     res["pow2"]["loss_options"][distname]["pow"] = 2
#     return res


for option in optionslist:
    new_exp_list = []
    for exp in exp_list:
        res_dict = option(exp)
        if len(res_dict.keys()) == 0:
            new_exp_list.append(exp)
        else:
            for new_tag, new_conf in res_dict.items():
                new_exp_list.append(
                    ExperimentConfig(new_conf, tags=exp.tags + [new_tag])
                )
    exp_list = new_exp_list

for exp in exp_list:
    print("_".join(exp.tags))
    print(exp.config)
    folder = "wd/" + "_".join(exp.tags)
    if not isdir(folder):
        mkdir(folder)
    OmegaConf.save(exp.config, f"{folder}/config.yaml")
print(
    "./run.sh remote --tag "
    + ",".join(["_".join(exp.tags) for exp in exp_list])
    + " train"
)
