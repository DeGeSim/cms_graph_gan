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
                    "optim": {
                        "params": {
                            "lr": 2.0e-4,
                        }
                    },
                },
                "disc": {
                    "name": "disc_fake",
                    "optim": {
                        "params": {
                            "lr": 2.0e-4,
                        }
                    },
                },
            },
        }
    ),
    ["jetnet"],
)

exp_list: List[ExperimentConfig] = [base_config]

optionslist: List[Callable] = []


def add_option(option: Callable):
    optionslist.append(option)


@add_option
def option_disc(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["dmp"]["models"]["disc"]["name"] = "disc_mp"
    res["dpnet"]["models"]["disc"]["name"] = "disc_pointnetmix"
    res["dpcgan"]["models"]["disc"]["name"] = "disc_pcgan"
    return res


@add_option
def option_disc_lr(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    # res["lrddef"]["models"]["disc"]["optim"]["params"]["lr"] = 2.0e-4
    res["lrdlow"]["models"]["disc"]["optim"]["params"]["lr"] = 2.0e-5
    res["lrdvlow"]["models"]["disc"]["optim"]["params"]["lr"] = 2.0e-6
    return res


@add_option
def option_gen_lr(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["lrgdef"]["models"]["gen"]["optim"]["params"]["lr"] = 2.0e-4
    res["lrglow"]["models"]["gen"]["optim"]["params"]["lr"] = 6.0e-5
    # res["lrgvlow"]["models"]["gen"]["optim"]["params"]["lr"] = 1.0e-5
    return res


@add_option
def option_optimizer(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    # res["Adamnodecay"]
    res["Adam"]["models"]["gen"]["optim"]["params"]["weight_decay"] = 1.0e-4
    res["Adam"]["models"]["disc"]["optim"]["params"]["weight_decay"] = 1.0e-4
    res["RMSprop"]["models"]["gen"]["optim"]["name"] = "RMSprop"
    res["RMSprop"]["models"]["disc"]["optim"]["name"] = "RMSprop"
    return res


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
