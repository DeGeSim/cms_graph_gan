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
loader_name: moons
comet_project_name: moons2
training:
  gan_mode: MSE
  val:
    use_for_stopping:
    - w1x
    - w1y
    metrics:
      - ft_w1
      - aoc
models:
  gen:
    losses_list: ["mmdpc"]
  disc:
    name: disc_fake
    optim:
      name: FakeOptimizer
loss_options:
    cd:
        factor: 1.0
        lpnorm: 1
        batch_wise: False
        pow: 2
    dcd:
        factor: 1.0
        alpha: 1000
        lpnorm: 1
        batch_wise: False
        pow: 1
model_param_options:
    gen_deeptree:
        n_global: 0
        branching_param:
            residual: True
            final_linear: False
            norm: 'none'
        ancestor_mpl:
            n_mpl: 0
            conv_name: GINConv
            skip_connecton: False
        child_mpl:
            n_mpl: 0
            conv_name: GINConv
            skip_connecton: False
ffn:
    activation: LeakyReLU
    hidden_layer_size: 512
    n_layers: 3
    norm: layernorm
    dropout: True

    """
    ),
    ["twomoons"],
)

exp_list: List[ExperimentConfig] = [base_config]

optionslist: List[Callable] = []


def add_option(option: Callable):
    optionslist.append(option)


@add_option
def option_dist(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["dcd"]["models"]["gen"]["losses_list"] = ["dcd"]
    res["cd"]["models"]["gen"]["losses_list"] = ["cd"]
    res["mmdpc"]["models"]["gen"]["losses_list"] = ["mmdpc"]
    return res


@add_option
def option_norm(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    if "dcd" in exp_config.tags:
        dist = "dcd"
    elif "cd" in exp_config.tags:
        dist = "cd"
    else:
        return res
    res["lpnorm1"]["loss_options"][dist]["lpnorm"] = 1
    res["lpnorm2"]["loss_options"][dist]["lpnorm"] = 2
    return res


@add_option
def option_pow(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    if "dcd" in exp_config.tags:
        dist = "dcd"
    elif "cd" in exp_config.tags:
        dist = "cd"
    else:
        return res
    res["pow1"]["loss_options"][dist]["pow"] = 1
    res["pow2"]["loss_options"][dist]["pow"] = 2
    return res


@add_option
def option_alpha(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    if "dcd" in exp_config.tags:
        dist = "dcd"
    else:
        return res
    res["alpha1"]["loss_options"][dist]["alpha1"] = 1
    res["alpha10"]["loss_options"][dist]["alpha1"] = 10
    res["alpha50"]["loss_options"][dist]["alpha1"] = 50
    res["alpha1000"]["loss_options"][dist]["alpha1"] = 1000
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
    "./run.sh remote --tag "
    + ",".join(["_".join(exp.tags) for exp in exp_list])
    + " train"
)
