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
        """
comet_project_name: moons2
loader_name: moons
seed: 0
models:
  gen:
    losses_list:
    - dcd
  disc:
    name: disc_fake
    params: {}
    optim:
      name: FakeOptimizer
training:
  val:
    interval: 100
    use_for_stopping:
    - w1x
    - w1y
    metrics:
    - ft_w1
    - aoc
optim_options:
    Adam:
        weight_decay: 1.0e-4
        lr: 1.e-4
    """
    ),
    ["twomoons_dist"],
)

exp_list: List[ExperimentConfig] = [base_config]

optionslist: List[Callable] = []


def add_option(option: Callable):
    if option.__name__ in [e.__name__ for e in optionslist]:
        raise Exception("dont add functions twice")
    optionslist.append(option)


@add_option
def option_lr(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    # res["f1"]["loss_options"]["dcd"]["factor"] = 1.0
    # res["f.1"]["loss_options"]["dcd"]["factor"] = 0.1
    res["lrm4"]["optim_options"]["Adam"]["lr"] = 1.0e-4
    res["lrm5"]["optim_options"]["Adam"]["lr"] = 1.0e-5
    res["lrm6"]["optim_options"]["Adam"]["lr"] = 1.0e-6
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
    OmegaConf.save(exp.config, f"{folder}/conf.yaml")
print(
    "./run.sh --tag "
    + ",".join(["_".join(exp.tags) for exp in exp_list])
    + " setup"
)
print(
    "./run.sh --tag "
    + ",".join(["_".join(exp.tags) for exp in exp_list])
    + " --remote train"
)

# @add_option
# def option_reg(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["nonorm"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "norm"
#     ] = "none"
#     res["batchnorm"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "norm"
#     ] = "batchnorm"
#     res["layernorm"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "norm"
#     ] = "layernorm"
#     return res

# @add_option
# def option_lpnorm(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["lp.5"]["loss_options"]["dcd"]["lpnorm"] = 0.5
#     res["lp1"]["loss_options"]["dcd"]["lpnorm"] = 1.0
#     res["lp2"]["loss_options"]["dcd"]["lpnorm"] = 2.0
#     return res


# @add_option
# def option_pow(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["pow.5"]["loss_options"]["dcd"]["pow"] = 0.5
#     res["pow1"]["loss_options"]["dcd"]["pow"] = 1.0
#     res["pow2"]["loss_options"]["dcd"]["pow"] = 2.0
#     return res


# @add_option
# def option_activation(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["ReLU"]["ffn"]["activation"] = "ReLU"
#     res["LeakyReLU"]["ffn"]["activation"] = "LeakyReLU"
#     return res


# @add_option
# def option_dropout(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["dropout"]["ffn"]["dropout"] = True
#     res["nodropout"]["ffn"]["dropout"] = False
#     return res


# @add_option
# def option_norm(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["nonorm"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "norm"
#     ] = "none"
#     res["batchnorm"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "norm"
#     ] = "batchnorm"
#     res["layernorm"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "norm"
#     ] = "layernorm"
#     return res
# @add_option
# def option_fl(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["fl"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "final_linear"
#     ] = True
#     res["nofl"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "final_linear"
#     ] = False
#     return res


# @add_option
# def option_norm(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["res"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "residual"
#     ] = True
#     res["nores"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "residual"
#     ] = False
#     return res
