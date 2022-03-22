import os
import random
from glob import glob

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fgsim.utils.cli import args
from fgsim.utils.oc_resolvers import register_resolvers
from fgsim.utils.oc_utils import gethash, removekeys

register_resolvers()


# Load the default settings, overwrite them
# witht the tag-specific settings and then
# overwrite those with cli arguments.
conf: DictConfig
if args.hash is not None:
    try:
        fn = glob(f"wd/*/{args.hash}/full_config.yaml")[0]
    except IndexError:
        raise IndexError("No such {args.hash} hash!")
    conf = OmegaConf.load(fn)
    conf["command"] = str(args.command)
else:
    with open("fgsim/default.yaml", "r") as fp:
        defaultconf = OmegaConf.load(fp)

    if args.tag != "default":
        fn = f"wd/{args.tag}/config.yaml"
        if not os.path.isfile(fn):
            raise FileNotFoundError
        with open(fn, "r") as fp:
            tagconf = OmegaConf.load(fp)
        conf = OmegaConf.unsafe_merge(defaultconf, tagconf, vars(args))

        # remove the dependency on hash and loader hash to be able to resolve
        conf_without_paths = removekeys(
            conf,
            [
                "path",
            ],
        )
        OmegaConf.resolve(conf_without_paths)

        # Compute a loader_hash
        # this hash will be part of where the preprocessed
        # dataset is safed to ensure the parameters dont change
        # Exclude the keys that do not affect the training
        exclude_keys = ["preprocess_training", "debug"] + [
            x for x in conf["loader"] if "num_workers" in x
        ]

        loader_params = removekeys(conf_without_paths["loader"], exclude_keys)
        conf["loader_hash"] = gethash(loader_params)

        hyperparameters = removekeys(
            conf_without_paths,
            [
                "command",
                "debug",
                "loglevel",
                "path",
            ]
            + [key for key in conf.keys() if key.endswith("_options")],
        )

        conf["hash"] = gethash(hyperparameters)
        # Infer the parameters here
        OmegaConf.resolve(conf)
    else:
        conf = OmegaConf.unsafe_merge(defaultconf, vars(args))


torch.manual_seed(conf.seed)
np.random.seed(conf.seed)
random.seed(conf.seed)

# Select the CPU/GPU
if torch.cuda.is_available() and conf["command"] in ["train", "test"]:
    device = torch.device("cuda:" + str(torch.cuda.device_count() - 1))
else:
    device = torch.device("cpu")
