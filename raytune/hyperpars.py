from ray import tune

hyperpars = {
    # "models": {
    #     "disc": {
    #         "optim": {
    #             "params": {
    #                 "lr": tune.loguniform(1e-6, 1e-3),
    #                 "betas": tune.choice([[0.9, 0.999], [0.0, 0.9]]),
    #             }
    #         },
    #     },
    # },
    # "training": {
    #     "disc_steps_per_gen_step": tune.randint(1, 2),
    # },
    "model_param_options": {
        "disc_deeptree": {
            "nodes": [30, 6, 1],
            "features": [3, 3, 20, 40],
            # "ffn_param": {
            # "n_layers": tune.randint(2, 5),
            # "hidden_layer_size": tune.randint(30, 100),
            # "norm": tune.choice(
            #     ["batchnorm", "layernorm", "spectral", "weight"]
            # ),
            # },
            "emb_param": {"n_ftx_latent": tune.randint(3, 30)},
            "bipart_param": {"n_heads": tune.randint(1, 20)},
            # "critics_param": {
            #     "n_ftx_latent": tune.randint(3, 30),
            #     "n_ftx_global": tune.randint(3, 30),
            #     "n_updates": tune.randint(1, 10),
            # },
        },
        "gen_deeptree": {
            # "n_global": tune.randint(0, 10),
            "pruning": tune.choice(["cut", "topk"]),
            "equivar": tune.choice([True, False]),
        },
    },
    # "tree_width": tune.choice(["wide", "slim"]),
    # "root_node_size": tune.randint(24, 1024),
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
