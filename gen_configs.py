from dataclasses import dataclass
from os import mkdir
from typing import Dict, List

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
    config=OmegaConf.create(
        """models:
    gen:
        name: gen_deeptree_pc
        losses_list: [WGenLoss]
    disc:
        losses_list: [WDiscLoss,GradientPenalty]
model_param_options:
    gen_deeptree_pc:
        n_hidden_features: 4
        n_global: 6
        n_branches: 2
        n_levels: 12
        post_gen_mp_steps: 10
        conv_name: GINConv
    disc_treepc:
        activation: Identity"""
    ),
    tags=[],
)
empty_dict_config = OmegaConf.create("")


def option_ce(input_conf: DictConfig) -> Dict[str, DictConfig]:
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["losses_list"].remove("WGenLoss")
    mod_conf["models"]["gen"]["losses_list"].append("CEGenLoss")
    mod_conf["models"]["disc"]["losses_list"].remove("WDiscLoss")
    mod_conf["models"]["disc"]["losses_list"].remove("GradientPenalty")
    mod_conf["models"]["disc"]["losses_list"].append("CEDiscLoss")
    mod_conf["model_param_options"]["disc_treepc"]["activation"] = "Sigmoid"
    return {"wd": input_conf, "ce": mod_conf}


def option_physics_loss(input_conf: DictConfig) -> Dict[str, DictConfig]:
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["losses_list"].append("physics")
    return {"nopl": input_conf, "pl": mod_conf}


def option_mean_dist_loss(input_conf: DictConfig) -> Dict[str, DictConfig]:
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["losses_list"].append("mean_dist")
    return {"nomd": input_conf, "md": mod_conf}


def option_wide(input_conf: DictConfig) -> Dict[str, DictConfig]:
    mod_conf = input_conf.copy()
    if "params" not in mod_conf["models"]["gen"]:
        mod_conf["models"]["gen"]["params"] = {}
    if "params" not in input_conf["models"]["gen"]:
        input_conf["models"]["gen"]["params"] = {}
    mod_conf["models"]["gen"]["params"]["n_branches"] = 4
    mod_conf["models"]["gen"]["params"]["n_levels"] = 7
    return {"slim": input_conf, "wide": mod_conf}


def option_conv(input_conf: DictConfig) -> Dict[str, DictConfig]:
    mod_conf = input_conf.copy()
    mod_conf["models"]["gen"]["params"]["conv_name"] = "GINConv"
    input_conf["models"]["gen"]["params"]["conv_name"] = "AncestorConv"
    # return { }
    return {"ancconv": input_conf, "gin": mod_conf}


exp_list: List[ExperimentConfig] = [base_config]
for option in [
    option_ce,
    option_physics_loss,
    option_mean_dist_loss,
    option_wide,
    option_conv,
]:
    new_exp_list = []
    for exp in exp_list:
        for new_tag, new_conf in option(exp.config).items():
            new_exp_list.append(
                ExperimentConfig(new_conf, tags=exp.tags + [new_tag])
            )
    exp_list = new_exp_list


for exp in exp_list:
    print(exp.tags)
    print(exp.config)
    folder = "wd/deeptree_" + "_".join(exp.tags)
    if not isdir(folder):
        mkdir(folder)
    OmegaConf.save(exp.config, f"{folder}/config.yaml")

# WGenLoss ->
