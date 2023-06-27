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
  disc:
    name: disc_deeptree
    additional_losses_list: ["histd"]
    optim:
      name: Adam
    scheduler:
      name: NullScheduler
  gen:
    additional_losses_list: ["feature_matching", "histg"]
    scheduler:
      name: NullScheduler
model_param_options:
  gen_deeptree:
    pruning: cut
    dim_red_in_branching: True
    n_global: 10
    branching_param:
      mode: "mat"
      residual: True
      final_linear: True
      norm: batchnorm
      res_mean: False
      res_final_layer: True
    connect_all_ancestors: True
    ancestor_mpl:
      n_mpl: 1
      n_hidden_nodes: 100
      conv_name: GINConv
      skip_connecton: True
    final_layer_scaler: False
  disc_deeptree:
      ffn_param:
        bias: false
        n_layers: 2
        hidden_layer_size: 40
        dropout: 0.0
        norm: spectral
      emb_param:
        n_ftx_latent: 10
      bipart_param:
        n_heads: 4
      critics_param:
        n_ftx_latent: 4
        n_ftx_global: 5
        n_updates: 2
training:
  gan_mode: Hinge
optim_options:
  gen:
    Adam:
      lr: 1.0e-05
      weight_decay: 1.0e-4
      betas: [0.9, 0.999]
  disc:
    Adam:
      lr: 3.0e-5
      weight_decay: 1.0e-4
      betas: [0.9, 0.999]
loss_options:
  feature_matching:
    factor: 1.0e-1
  histd:
    factor: 1.0e-01
  histg:
    factor: 1.0e-01
    """
    ),
    ["scan"],
)


optionslist: List[Callable] = []


def add_option(option: Callable):
    if option.__name__ in [e.__name__ for e in optionslist]:
        raise Exception("dont add functions twice")
    optionslist.append(option)


exp_list: List[ExperimentConfig] = [base_config]


# equivar vs regular


# MSE vs Hinge
# @add_option
# def option_ganmode(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["W"]["training"]["gan_mode"] = "W"
#     return res


# @add_option
# def option_hist(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["gnoh"]["models"]["gen"]["additional_losses_list"] = ["feature_matching"]
#     res["dnoh"]["models"]["disc"]["additional_losses_list"] = []
#     res["noh"]["models"]["gen"]["additional_losses_list"] = ["feature_matching"]
#     res["noh"]["models"]["disc"]["additional_losses_list"] = []
#     return res


# @add_option
# def option_gnorm(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["gbwn"]["ffn"]["norm"] = "bwn"
#     res["gwn"]["ffn"]["norm"] = "weight"
#     res["gln"]["ffn"]["norm"] = "layernorm"
#     res["gsn"]["ffn"]["norm"] = "spectral"
#     return res


# @add_option
# def option_cnorm(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)

#     res["cbn"]["model_param_options"]["disc_deeptree"]["ffn_param"][
#         "norm"
#     ] = "batchnorm"
#     res["cbn"]["model_param_options"]["disc_deeptree"]["cnu_param"][
#         "norm"
#     ] = "batchnorm"

#     res["cbwn"]["model_param_options"]["disc_deeptree"]["ffn_param"]["norm"
# ] = "bwn"
#     res["cbwn"]["model_param_options"]["disc_deeptree"]["cnu_param"]["norm"
# ] = "bwn"

#     res["cln"]["model_param_options"]["disc_deeptree"]["ffn_param"][
#         "norm"
#     ] = "layernorm"
#     res["cln"]["model_param_options"]["disc_deeptree"]["cnu_param"][
#         "norm"
#     ] = "layernorm"

#     res["cfsn"]["model_param_options"]["disc_deeptree"]["emb_param"][
#         "norm"
#     ] = "spectral"

#     return res


@add_option
def option_branching(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)

    # topk
    res["tkmat"]["model_param_options"]["gen_deeptree"]["pruning"] = "topk"
    res["tkmat"]["model_param_options"]["gen_deeptree"]["branching_param"][
        "mode"
    ] = "mat"

    res["tkeqv"]["model_param_options"]["gen_deeptree"]["pruning"] = "topk"
    res["tkeqv"]["model_param_options"]["gen_deeptree"]["branching_param"][
        "mode"
    ] = "equivar"

    res["tknoise"]["model_param_options"]["gen_deeptree"]["pruning"] = "topk"
    res["tknoise"]["model_param_options"]["gen_deeptree"]["branching_param"][
        "mode"
    ] = "noise"

    # cut

    res["cuteqv"]["model_param_options"]["gen_deeptree"]["branching_param"][
        "mode"
    ] = "equivar"

    res["cutnoise"]["model_param_options"]["gen_deeptree"]["branching_param"][
        "mode"
    ] = "noise"

    return res


@add_option
def option_dropout(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)

    res["d.2"]["model_param_options"]["disc_deeptree"]["ffn_param"]["bias"] = 0.2
    res["d.5"]["model_param_options"]["disc_deeptree"]["ffn_param"]["bias"] = 0.5

    return res


@add_option
def option_nglob(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)

    res["gn0glob"]["model_param_options"]["gen_deeptree"]["n_global"] = 0
    res["gn40glob"]["model_param_options"]["gen_deeptree"]["n_global"] = 40

    return res


@add_option
def option_gnoac(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)

    res["g0ac"]["model_param_options"]["gen_deeptree"]["ancestor_mpl"]["n_mpl"] = 0
    res["g3ac"]["model_param_options"]["gen_deeptree"]["ancestor_mpl"]["n_mpl"] = 3

    return res


@add_option
def option_gacgat(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
    res = defaultdict(exp_config.config.copy)

    res["GATv2MinConv"]["model_param_options"]["gen_deeptree"]["ancestor_mpl"][
        "conv_name"
    ] = "GATv2MinConv"
    return res


# @add_option
# def option_beta(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["dnomom"]["optim_options"]["disc"]["Adam"]["betas"] = [0.0, 0.9]
#     res["dSGD"]["models"]["disc"]["optim"]["name"] = "SGD"
#     return res


# @add_option
# def option_swa(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)
#     res["gCyclicLR"]["models"]["gen"]["scheduler"]["name"] = "CyclicLR"
#     res["gSWA"]["models"]["gen"]["scheduler"]["name"] = "SWA"
#     return res


# @add_option
# def option_glr(exp_config: ExperimentConfig) -> Dict[str, DictConfig]:
#     res = defaultdict(exp_config.config.copy)

#     res["glrhigh"]["optim_options"]["gen"]["Adam"]["lr"] *= 5
#     res["glrlow"]["optim_options"]["gen"]["Adam"]["lr"] /= 5

#     res["dlrhigh"]["optim_options"]["disc"]["Adam"]["lr"] *= 5
#     res["dlrlow"]["optim_options"]["disc"]["Adam"]["lr"] /= 5
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


if False:
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
    "./scripts/run.sh --tag "
    + ",".join(["_".join(exp.tags) for exp in exp_list])
    + " setup"
)
print(
    "./scripts/run.sh --tag "
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
