from pathlib import Path

from omegaconf import OmegaConf

run_name = "jn150-ddt-tune1"
rayconf = OmegaConf.load(Path(f"~/fgsim/wd/ray/{run_name}/conf.yaml").expanduser())


def process_exp_config(exp_config):
    # manipulate the config for the tree
    # wide = exp_config["tree_width"] == "wide"
    # root_size = exp_config["root_node_size"]
    # exp_config["tree"] = {}
    # if wide:
    #     exp_config.tree["branches"] = [3, 10]
    #     scaling = np.power(root_size / 3.0, 1 / 2)
    #     exp_config.tree["features"] = [root_size, int(root_size / scaling), 3]

    # else:
    #     exp_config.tree["branches"] = [2, 3, 5]
    #     scaling = np.power(root_size / 3.0, 1 / 3)
    #     exp_config.tree["features"] = [
    #         root_size,
    #         int(root_size / scaling),
    #         int(root_size / scaling**2),
    #         3,
    #     ]

    # del exp_config["tree_width"]
    # del exp_config["root_node_size"]

    return exp_config
