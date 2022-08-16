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
models:
    gen:
        losses_list: [CEGenLoss, cd]
    disc:
        losses_list: [CEDiscLoss]
model_param_options:
    gen_deeptree:
        n_global: 5
        conv_parem:
            add_self_loops: True
            msg_nn_bool: True
            upd_nn_bool: True
            msg_nn_include_edge_attr: False
            msg_nn_include_global: True
            upd_nn_include_global: True
            residual: True
        branching_param:
            residual: True
        child_param:
            n_mpl: 3
            n_hidden_nodes: 128
ffn:
    activation: LeakyReLU
    hidden_layer_size: 128
    n_layers: 3
    activation_params:
        negative_slope: 0.1
    init_weights_bias_const: 0.1
    init_weights: xavier_uniform_
    normalize: True
tree:
    branches: [2, 3, 5]
    features: [256, 64, 32, 3]
    """
    ),
    ["jetnet"],
)

exp_list: List[ExperimentConfig] = [base_config]

optionslist: List[Callable] = []


def add_option(option: Callable):
    optionslist.append(option)


# Model type
@add_option
def option_gantype(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["CE"]
    res["W"]["models"]["gen"]["losses_list"] = ["WGenLoss", "cd"]
    res["W"]["models"]["disc"]["losses_list"] = [
        "WDiscLoss",
        "GradientPenalty",
    ]
    res["MSE"]["models"]["gen"]["losses_list"] = ["MSEGenLoss", "cd"]
    res["MSE"]["models"]["disc"]["losses_list"] = ["MSEDiscLoss"]

    res["benno"]["models"]["gen"]["losses_list"] = ["MSEGenLoss"]
    res["benno"]["models"]["disc"]["losses_list"] = ["MSEDiscLoss"]
    return res


# FFN
@add_option
def option_ffnlayer(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    # res["ffnlayer3"]["ffn"]["n_layers"] = 3
    res["ffnlayer5"]["ffn"]["n_layers"] = 5
    res["ffnlayer10"]["ffn"]["n_layers"] = 10
    return res


@add_option
def option_ffnnodes(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    # res["ffnodes128"]["ffn"]["hidden_layer_size"] = 128
    res["ffnnodes512"]["ffn"]["hidden_layer_size"] = 512
    res["ffnnodes1024"]["ffn"]["hidden_layer_size"] = 1024
    return res


@add_option
def option_ffninit(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["glurot"]["ffn"]["init_weights"] = "xavier_uniform_"
    # res["kaiming"]["ffn"]["init_weights"] = "kaiming_uniform_"

    return res


# deep tree gen
@add_option
def option_n_global(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["nglobal0"]["model_param_options"]["gen_deeptree"]["n_global"] = 0
    # res["nglobal3"]["model_param_options"]["gen_deeptree"]["n_global"] = 3
    # res["nglobal7"]["model_param_options"]["gen_deeptree"]["n_global"] = 7
    return res


# conv_parem
@add_option
def option_selfloops(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["selfloops"]["model_param_options"]["gen_deeptree"]["conv_parem"][
        "add_self_loops"
    ] = True
    # res["¬selfloops"]["model_param_options"]["gen_deeptree"]["conv_parem"][
    #     "add_self_loops"
    # ] = False
    return res


@add_option
def option_msgnn(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["msgnn"]["model_param_options"]["gen_deeptree"]["conv_parem"][
        "msg_nn_bool"
    ] = True
    # res["¬msgnn"]["model_param_options"]["gen_deeptree"]["conv_parem"][
    #     "msg_nn_bool"
    # ] = False
    return res


@add_option
def option_updnn(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["updnn"]["model_param_options"]["gen_deeptree"]["conv_parem"][
        "upd_nn_bool"
    ] = True
    # res["¬updnn"]["model_param_options"]["gen_deeptree"]["conv_parem"][
    #     "upd_nn_bool"
    # ] = False
    return res


@add_option
def option_msgea(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    # res["msgea"]["model_param_options"]["gen_deeptree"]["conv_parem"][
    #     "msg_nn_include_edge_attr"
    # ] = True
    res["¬msgea"]["model_param_options"]["gen_deeptree"]["conv_parem"][
        "msg_nn_include_edge_attr"
    ] = False
    return res


@add_option
def option_msgg(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["msgg"]["model_param_options"]["gen_deeptree"]["conv_parem"][
        "msg_nn_include_global"
    ] = True
    # res["¬msgg"]["model_param_options"]["gen_deeptree"]["conv_parem"][
    #     "msg_nn_include_global"
    # ] = False
    return res


@add_option
def option_updg(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["updg"]["model_param_options"]["gen_deeptree"]["conv_parem"][
        "upd_nn_include_global"
    ] = True
    # res["¬updg"]["model_param_options"]["gen_deeptree"]["conv_parem"][
    #     "upd_nn_include_global"
    # ] = False
    return res


@add_option
def option_deepconvres(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["convres"]["model_param_options"]["gen_deeptree"]["conv_parem"][
        "residual"
    ] = True
    # res["¬convres"]["model_param_options"]["gen_deeptree"]["conv_parem"][
    #     "residual"
    # ] = False
    return res


# Child param
@add_option
def option_cconvlayer(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["cconvlayer2"]["model_param_options"]["gen_deeptree"]["child_param"][
        "n_mpl"
    ] = 2
    res["cconvlayer4"]["model_param_options"]["gen_deeptree"]["child_param"][
        "n_mpl"
    ] = 4
    return res


@add_option
def option_cconvnodes(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    # res["cconvnodes128"]["model_param_options"]["gen_deeptree"]["child_param"][
    #     "n_mpl"
    # ] = 128
    res["cconvnodes512"]["model_param_options"]["gen_deeptree"]["child_param"][
        "n_mpl"
    ] = 512
    res["cconvnodes1024"]["model_param_options"]["gen_deeptree"]["child_param"][
        "n_mpl"
    ] = 1024
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
