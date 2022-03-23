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
    return {"ce": mod_conf}  # ,"wd": input_conf,


@add_option
def option_gen(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    out_dict = {}
    generators = ["gen_deeptree", "gen_edgeconv"]  # "gen_treepc",,
    for gen_name in generators:
        mod_conf = input_conf.copy()
        mod_conf["models"]["gen"]["name"] = gen_name
        out_dict[gen_name.replace("gen_", "G")] = mod_conf
    return out_dict


# Option for the generator
# @add_option
def option_conv(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    if exp_config.config["models"]["gen"]["name"] != "gen_deeptree":
        return {}
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["params"]["conv_name"] = "GINConv"
    mod_conf = input_conf.copy()

    return {"ancconv": input_conf, "gin": mod_conf}


@add_option
def option_branching_res(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    if exp_config.config["models"]["gen"]["name"] != "gen_deeptree":
        return {}
    brres_conf = exp_config.config.copy()
    brres_conf["layer_options"]["BranchingLayer"]["residual"] = True
    brresno_conf = exp_config.config.copy()
    brresno_conf["layer_options"]["BranchingLayer"]["residual"] = False
    return {"brres": brres_conf, "brresno": brresno_conf}


@add_option
def option_ancestorconv_res(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    if exp_config.config["models"]["gen"]["name"] != "gen_deeptree":
        return {}
    if (
        exp_config.config["model_param_options"]["gen_deeptree"]["conv_name"]
        != "AncestorConv"
    ):
        return {}
    acres_conf = exp_config.config.copy()
    acres_conf["layer_options"]["AncestorConv"]["residual"] = True
    acresno_conf = exp_config.config.copy()
    acresno_conf["layer_options"]["AncestorConv"]["residual"] = False
    return {"acres": acres_conf, "acresno": acresno_conf}


# @add_option
def option_disc(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    out_dict = {}
    discriminators = ["disc_graphgym"]  # "disc_hlvs", "disc_treepc"
    for disc_name in discriminators:
        mod_conf = input_conf.copy()
        mod_conf["models"]["disc"]["name"] = disc_name
        out_dict[disc_name.replace("disc_", "D")] = mod_conf
    return out_dict


# FFN config
@add_option
def option_ffn_activation(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    mod_conf["ffn"]["activation"] = "LeakyReLU"
    mod_conf["ffn"]["activation_params"] = {"negative_slope": 0.2}
    return {"selu": input_conf, "leakyrelu": mod_conf}


# Losses
# @add_option
def option_physics_loss(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["losses_list"].append("physics")
    return {"nopl": input_conf, "pl": mod_conf}  #


@add_option
def option_mean_dist_loss(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["losses_list"].append("mean_dist")
    return {"md": mod_conf, "nomd": input_conf}


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
