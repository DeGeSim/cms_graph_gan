from dataclasses import dataclass
from os import mkdir
from typing import Callable, Dict, List

from genericpath import isdir
from omegaconf import DictConfig, OmegaConf

from fgsim.utils.oc_resolvers import register_resolvers
from fgsim.utils.oc_utils import compute_update_config, removekeys

register_resolvers()


@dataclass
class ExperimentConfig:
    config: DictConfig
    tags: List[str]


@dataclass
class ExperimentOption:
    config_change: Dict[str, DictConfig]


with open("fgsim/default.yaml", "r") as fp:
    default_config: DictConfig = OmegaConf.load(fp)

default_config = removekeys(
    default_config,
    [
        "command",
        "debug",
        "loglevel",
        "path",
    ],
)

base_config = ExperimentConfig(default_config.copy(), [])

exp_list: List[ExperimentConfig] = [base_config]

optionslist: List[Callable] = []


def add_option(option: Callable):
    optionslist.append(option)


@add_option
def option_gan(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["losses_list"].remove("WGenLoss")
    mod_conf["models"]["gen"]["losses_list"].append("CEGenLoss")
    mod_conf["models"]["disc"]["losses_list"].remove("WDiscLoss")
    mod_conf["models"]["disc"]["losses_list"].remove("GradientPenalty")
    mod_conf["models"]["disc"]["losses_list"].append("CEDiscLoss")
    return {"wd": input_conf, "ce": mod_conf}  # ,


@add_option
def option_gen(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    out_dict = {}
    generators = ["gen_edgeconv", "gen_deeptree"]  # "gen_treepc",
    for gen_name in generators:
        mod_conf = input_conf.copy()
        mod_conf["models"]["gen"]["name"] = gen_name
        out_dict[gen_name.replace("gen_", "G")] = mod_conf
    return out_dict


@add_option
def option_disc(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    out_dict = {}
    discriminators = ["disc_graphgym"]  # "disc_hlvs", "disc_treepc"
    for disc_name in discriminators:
        mod_conf = input_conf.copy()
        mod_conf["models"]["disc"]["name"] = disc_name
        out_dict[disc_name.replace("disc_", "D")] = mod_conf
    return out_dict


# @add_option
def option_physics_loss(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["losses_list"].append("physics")
    return {"nopl": input_conf, "pl": mod_conf}  #


# @add_option
def option_mean_dist_loss(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["losses_list"].append("mean_dist")
    return {"nomd": input_conf, "md": mod_conf}  #


@add_option
def option_conv(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    if "gen_deeptree" not in exp_config.tags:
        return {}
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["params"]["conv_name"] = "GINConv"
    mod_conf = input_conf.copy()

    return {"ancconv": input_conf, "gin": mod_conf}  #


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
    exp.config = compute_update_config(
        default_config,
        exp.config,
    )


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
