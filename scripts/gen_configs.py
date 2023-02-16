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
comet_project_name: jetnet_uc
dataset_name: jetnet
models:
    gen:
        additional_losses_list: []
        optim:
            name: Adam
        scheduler:
            name: CyclicLR
        retain_graph_on_backprop: True
    disc:
        name: disc_benno_lin
        optim:
            name: Adam
        scheduler:
            name: NullScheduler
training:
    gan_mode: MSE
optim_options:
  disc:
    Adam:
      betas: [0.0, 0.9]
tree:
  branches: [2,3,5]
  features: [64,33,20,3]
dataset_options:
  jetnet:
    loader:
      dataset_glob: "**/q.hdf5"
model_param_options:
    gen_deeptree:
        n_cond: 1
        pruning: "cut"
        equivar: False
    """
    ),
    ["uc", "q"],
)


optionslist: List[Callable] = []


def add_option(option: Callable):
    if option.__name__ in [e.__name__ for e in optionslist]:
        raise Exception("dont add functions twice")
    optionslist.append(option)


exp_list: List[ExperimentConfig] = [base_config]


# equivar vs regular


# MSE vs Hinge
@add_option
def option_ganmode(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["MSE"]
    res["Hinge"]["training"]["gan_mode"] = "Hinge"
    return res


def option_ext0(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["noext0"]
    res["ext0"]["models"]["gen"]["additional_losses_list"] = ["extozero"]
    return res


@add_option
def option_swa(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["SWA"]["models"]["gen"]["scheduler"]["name"] = "SWA"
    res["SWA"]["models"]["disc"]["scheduler"]["name"] = "SWA"
    res["noSWA"]
    return res


@add_option
def option_prune(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["cut"]["model_param_options"]["gen_deeptree"]["pruning"] = "cut"
    res["topk"]["model_param_options"]["gen_deeptree"]["pruning"] = "topk"
    return res


@add_option
def option_eqv(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)
    res["eqv"]["model_param_options"]["gen_deeptree"]["equivar"] = True
    res["noeqv"]["model_param_options"]["gen_deeptree"]["equivar"] = False
    return res


# @add_option
# def option_beta(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["dbeta1"]["optim_options"]["disc"]["Adam"]["betas"] = [0.9, 0.999]
#     res["dbeta2"]["optim_options"]["disc"]["Adam"]["betas"] = [0.0, 0.9]
#     return res


# @add_option
# def option_maxschedlr(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     scheduler = exp_config.config["models"]["gen"]["scheduler"]["name"]
#     if scheduler == "NullScheduler":
#         return {}
#     res = defaultdict(exp_config.config.copy)
#     res["lrf100"]["scheduler_options"][scheduler]["max_lr_factor"] = 100
#     # res["lrf25"]["scheduler_options"][scheduler]["max_lr_factor"] = 25
#     res["lrf10"]["scheduler_options"][scheduler]["max_lr_factor"] = 10
#     return res


if True:
    # factorize all options
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
else:
    # Only apply one option at the time
    for option in optionslist:
        res_dict = option(base_config)
        for new_tag, new_conf in res_dict.items():
            if new_conf == base_config.config:
                raise Exception(f"Tag {new_tag} is the same as base config")
            exp_list.append(
                ExperimentConfig(new_conf, tags=base_config.tags + [new_tag])
            )


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
# def option_activation(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     # res["LeakyReLu"]["ffn"]["activation"] = "LeakyReLU"
#     res["Tanh"]["ffn"]["activation"] = "Tanh"
#     res["SELU"]["ffn"]["activation"] = "SELU"
#     return res


# @add_option
# def option_tree(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     # res["f1"]["loss_options"]["dcd"]["factor"] = 1.0
#     # res["f.1"]["loss_options"]["dcd"]["factor"] = 0.1
#     # res["4lvl"]
#     res["8lvl"]["tree"]["branches"] = [2,3,5]
#     res["8lvl"]["tree"]["features"] = [256, 128, 64, 32, 16, 8, 4, 3]
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
#     # res["fl"]["model_param_options"]["gen_deeptree"]["branching_param"][
#     #     "final_linear"
#     # ] = True
#     res["nofl"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "residual"
#     ] = False
#     res["nofl"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "final_linear"
#     ] = False
#     return res


# @add_option
# def option_dim_red_in_branching(
#     exp_config: ExperimentConfig,
# ) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["direct"]["model_param_options"]["gen_deeptree"][
#         "dim_red_in_branching"
#     ] = True
#     # res["indirect"]["model_param_options"]["gen_deeptree"][
#     #     "dim_red_in_branching"
#     # ] = False
#     return res


# @add_option
# def option_res_mean(
#     exp_config: ExperimentConfig,
# ) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["resmean"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "res_mean"
#     ] = True
#     # res["resadd"]["model_param_options"]["gen_deeptree"]["branching_param"][
#     #     "res_mean"
#     # ] = False

#     return res


# @add_option
# def option_norm(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     # res["res"]["model_param_options"]["gen_deeptree"]["branching_param"][
#     #     "residual"
#     # ] = True
#     res["nores"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "residual"
#     ] = False
#     return res


# @add_option
# def option_res_final_layer(
#     exp_config: ExperimentConfig,
# ) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["flres"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "res_final_layer"
#     ] = True
#     # res["noflres"]["model_param_options"]["gen_deeptree"]["branching_param"][
#     #     "res_final_layer"
#     # ] = False

#     return res


# @add_option
# def option_reg(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["nonorm"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "norm"
#     ] = "none"
#     # res["batchnorm"]["model_param_options"]["gen_deeptree"]["branching_param"][
#     #     "norm"
#     # ] = "batchnorm"
#     res["layernorm"]["model_param_options"]["gen_deeptree"]["branching_param"][
#         "norm"
#     ] = "layernorm"
#     return res
