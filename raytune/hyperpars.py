from ray import tune

hyperpars = {
    "models": {
        "gen": {
            "optim": {"params": {"lr": tune.loguniform(1e-7, 1e-3)}},
        },
        "disc": {
            "optim": {"params": {"lr": tune.loguniform(1e-7, 1e-3)}},
            "name": tune.choice(["disc_mp", "disc_benno"]),
        },
    },
    "training": {
        "gan_mode": tune.choice(["CE", "W", "MSE"]),
        "disc_steps_per_gen_step": tune.randint(1, 10),
    },
    "layer_options": {
        "GINConv": {
            "final_linear": tune.choice([True, False]),
        },
        "GINCConv": {
            "final_linear": tune.choice([True, False]),
        },
        "DeepConv": {
            "add_self_loops": tune.choice([True, False]),
            "nns": tune.choice(["msg", "upd", "both"]),
            "msg_nn_include_edge_attr": tune.choice([True, False]),
            "msg_nn_include_global": tune.choice([True, False]),
            "msg_nn_final_linear": tune.choice([True, False]),
            "upd_nn_include_global": tune.choice([True, False]),
            "upd_nn_final_linear": tune.choice([True, False]),
            "residual": tune.choice([True, False]),
        },
    },
    "model_param_options": {
        "gen_deeptree": {
            "n_global": tune.randint(0, 10),
            "branching_param": {"residual": tune.choice([True, False])},
            "connect_all_ancestors": tune.choice([True, False]),
            "ancestor_mpl": {
                "n_mpl": tune.randint(1, 3),
                "n_hidden_nodes": tune.randint(20, 2048),
                "conv_name": tune.choice(["GINConv", "GINCConv", "DeepConv"]),
                "skip_connecton": tune.choice([True, False]),
            },
            "child_mpl": {
                "n_mpl": tune.randint(0, 3),
                "n_hidden_nodes": tune.randint(20, 2048),
                "conv_name": tune.choice(["GINConv", "GINCConv", "DeepConv"]),
                "skip_connecton": tune.choice([True, False]),
            },
            "final_layer_scaler": tune.choice([True, False]),
        }
    },
    "ffn": {
        "hidden_layer_size": tune.randint(20, 2048),
        "n_layers": tune.randint(1, 12),
        "norm": tune.choice(["batchnorm", "layernorm", "none"]),
    },
    "tree_width": tune.choice(["wide", "slim"]),
    "root_node_size": tune.randint(24, 1024),
}

# "tree": tune.choice(
#     [
#         {"branches": [2, 3, 5], "features": [256, 64, 32, 3]},
#         {"branches": [3, 10], "features": [256, 64, 3]},
#     ]
# ),
# "batchnorm": tune.choice([True, False]),
# "dropout": tune.choice([True, False]),
# "conv_param": {
#     # "add_self_loops": tune.choice([True, False]),
#     "nns": tune.choice(["both", "upd", "msg"]),
#     # "residual": tune.choice([True, False]),
# },
# # "branching_param": {"residual": tune.choice([True, False])},
# "child_param": {
#     "n_mpl": tune.randint(0, 5),
#     # "n_hidden_nodes": tune.choice([128, 512, 1024]),
# },

# hyperpars = {
#     "models": {
#         "gen": {
#             "additional_losses_list": [tune.choice(["cd", "dcd"])],
#             "optim": {"params": {"lr": tune.loguniform(1e-7, 1e-3)}},
#         },
#         "disc": {
#             "optim": {"params": {"lr": tune.loguniform(1e-7, 1e-3)}},
#         },
#     },
#     "training": {"gan_mode": tune.choice(["CE", "W", "MSE"])},
#     "model_param_options": {
#         "gen_deeptree": {
#             "n_global": tune.randint(0, 10),
#             "conv_param": {
#                 "add_self_loops": tune.choice([True, False]),
#                 "nns": tune.choice(["both", "upd", "msg"]),
#                 "msg_nn_include_edge_attr": False,
#                 "msg_nn_include_global": tune.choice([True, False]),
#                 "upd_nn_include_global": tune.choice([True, False]),
#                 "residual": tune.choice([True, False]),
#             },
#             "branching_param": {"residual": tune.choice([True, False])},
#             "child_param": {
#                 "n_mpl": tune.randint(0, 5),
#                 "n_hidden_nodes": tune.choice([128, 512, 1024]),
#             },
#         }
#     },
#     "ffn": {
#         "activation": tune.choice(["LeakyReLU", "ReLU", "SELU"]),
#         "hidden_layer_size": tune.choice([128, 256, 512]),
#         "n_layers": tune.choice([3, 5, 7]),
#         "init_weights": tune.choice(["xavier_uniform_", "kaiming_uniform_"]),
#         "normalize": tune.choice([True, False]),
#     },
#     "tree": tune.choice(
#         [
#             {"branches": [2, 3, 5], "features": [256, 64, 32, 3]},
#             {"branches": [3, 10], "features": [256, 64, 3]},
#         ]
#     ),
# }
