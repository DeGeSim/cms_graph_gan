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
                "gen": {"losses_list": ["mmdpc", "outside_interval"]},
                "disc": {"name": "disc_fake", "losses_list": []},
            }
        }
    ),
    [],
)

exp_list: List[ExperimentConfig] = [base_config]

optionslist: List[Callable] = []


def add_option(option: Callable):
    optionslist.append(option)


# %%


@add_option
def option_gen(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    input_conf["models"]["gen"]["name"] = "gen_deeptree"
    mod_conf["models"]["gen"]["name"] = "gen_linear"
    return {"gentd": input_conf, "genlin": mod_conf}


@add_option
def option_kernel(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    input_conf["loss_options"] = {"mmdpc": {"kernel": "rbf"}}
    mod_conf = input_conf.copy()
    mod_conf["loss_options"] = {"mmdpc": {"kernel": "multiscale"}}
    return {"rbf": input_conf, "multiscale": mod_conf}


@add_option
def option_bandwidth(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    input_conf = exp_config.config
    mod_conf = input_conf.copy()
    mod_conf2 = input_conf.copy()
    if input_conf["loss_options"]["mmdpc"]["kernel"] == "rbf":
        input_conf["loss_options"] = {"mmdpc": {"bandwidth": [0.01, 0.1, 2, 10]}}
        mod_conf["loss_options"] = {"mmdpc": {"bandwidth": [10, 15, 20, 50]}}
        mod_conf2["loss_options"] = {"mmdpc": {"bandwidth": [20, 40, 80, 150]}}
    elif input_conf["loss_options"]["mmdpc"]["kernel"] == "multiscale":
        input_conf["loss_options"] = {
            "mmdpc": {"bandwidth": [0.01, 0.05, 0.1, 1.3]}
        }
        mod_conf["loss_options"] = {"mmdpc": {"bandwidth": [0.2, 0.5, 0.9, 1.3]}}
        mod_conf2["loss_options"] = {"mmdpc": {"bandwidth": [1, 2, 5, 10]}}
    else:
        raise RuntimeError
    return {"bwfine": input_conf, "bwdefault": mod_conf, "bwcoarse": mod_conf2}


# %%

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


# for exp in exp_list:
#     exp.config = compute_update_config(
#         base_config.config,
#         exp.config,
#     )

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

# %%
