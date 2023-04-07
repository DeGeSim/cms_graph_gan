project_name = "jn150-ddt-tune2"


hyperpars = {
    "models": {
        "disc": {
            "optim": {
                "params": {
                    "lr": {
                        "distribution": "log_uniform",
                        "min": 1e-6,
                        "max": 1e-3,
                    },
                    "betas": {"values": [[0.9, 0.999], [0.0, 0.9]]},
                }
            }
        },
    },
    "training": {
        "disc_steps_per_gen_step": {"min": 1, "max": 6},
    },
    "model_param_options": {
        "disc_deeptree": {
            #         "nodes": [30, 6, 1],
            #         "features": [3, 3, 20, 40],
            "ffn_param": {
                "n_layers": {"min": 2, "max": 5},
                "hidden_layer_size": {"min": 30, "max": 100},
                "norm": {
                    "values": ["batchnorm", "layernorm", "spectral", "weight"]
                },
            },
            "emb_param": {"n_ftx_latent": {"min": 3, "max": 30}},
            "bipart_param": {"n_heads": {"min": 1, "max": 20}},
            "critics_param": {
                "n_ftx_latent": {"min": 3, "max": 30},
                "n_ftx_global": {"min": 3, "max": 30},
                "n_updates": {"min": 1, "max": 10},
            },
        },
        "gen_deeptree": {
            "n_global": {"min": 0, "max": 10},
            "pruning": {"values": ["cut", "topk"]},
            "equivar": {"values": [True, False]},
        },
    },
}


def wrap_recur(d):
    if not isinstance(d, dict) or set(d.keys()) in [
        {"values"},
        {"min", "max"},
        {"min", "max", "distribution"},
    ]:
        return d
    else:
        return {"parameters": {k: wrap_recur(v) for k, v in d.items()}}


def wrapped_hyperpars(d):
    return {k: wrap_recur(v) for k, v in d.items()}


hyperpars = wrapped_hyperpars(hyperpars)
