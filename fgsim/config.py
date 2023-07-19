import os
import random
from glob import glob
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf

from fgsim.utils.cli import get_args
from fgsim.utils.oc_resolvers import register_resolvers
from fgsim.utils.oc_utils import dict_to_keylist, gethash, removekeys

register_resolvers()


defaultconf = OmegaConf.load(Path("~/fgsim/fgsim/default.yaml").expanduser())
# Load the default settings, overwrite them
# witht the tag-specific settings and then
# overwrite those with cli arguments.
conf: DictConfig = defaultconf.copy()
hyperparameters: DictConfig = DictConfig({})


def compute_conf(default, *confs):
    default = default.copy()

    # Assert, that only keys existing in the default are overwritten
    default_key_set = set(dict_to_keylist(default))
    for c in confs:
        c_key_set = set(dict_to_keylist(c)) - {"/hash", "/debug", "/command"}
        if not c_key_set.issubset(default_key_set):
            raise Exception(
                "Key not present in the default config."
                f" Difference:\n{c_key_set - default_key_set}"
            )

    newconf = OmegaConf.merge(*(default, *confs))

    # remove the dependency on hash and loader hash to be able to resolve
    conf_without_paths = removekeys(
        newconf,
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
        x for x in newconf["loader"] if "n_workers" in x
    ]

    loader_params = removekeys(conf_without_paths["loader"], exclude_keys)
    newconf["loader_hash"] = gethash(loader_params)

    hyperparameters = removekeys(
        conf_without_paths,
        [
            "command",
            "debug",
            "loglevel",
            "loglevel_qf",
            "remote",
            "path",
            "project_name",
            "ray",
        ]
        + [key for key in newconf.keys() if key.endswith("_options")],
    )

    newconf["hash"] = gethash(hyperparameters)
    # Infer the parameters here
    OmegaConf.resolve(newconf)
    for k in newconf.path:
        newconf.path[k] = str(Path(newconf.path[k]).expanduser())
    # remove the options:
    for key in list(newconf.keys()):
        if key.endswith("_options"):
            del newconf[key]
    conf.update(newconf)
    return newconf, hyperparameters


def parse_arg_conf(args=None):
    if args is None:
        args = get_args()
    if args.hash is not None:
        try:
            if args.ray:
                globstr = f"wd/ray/*/{args.hash}/"
            else:
                globstr = f"wd/*/{args.hash}/"
            folder = Path(glob(globstr)[0])
            assert folder.is_dir()
        except IndexError:
            raise IndexError(f"No experiement with hash {args.hash} is set up.")
        conf = OmegaConf.load(folder / "conf.yaml")
        hyperparameters = OmegaConf.load(folder / "hyperparameters.yaml")

        conf["command"] = str(args.command)
    else:
        fn = f"wd/{args.tag}/conf.yaml"
        if os.path.isfile(fn):
            tagconf = OmegaConf.load(fn)
        else:
            if args.tag == "default":
                tagconf = OmegaConf.create({})
            else:
                raise FileNotFoundError(f"Tag {args.tag} has no conf.yaml file.")
        conf, hyperparameters = compute_conf(defaultconf, tagconf, vars(args))
    if conf.command in ["train", "test"]:
        setup_ml()
    return conf, hyperparameters


device = None


np.set_printoptions(formatter={"float_kind": "{:.3g}".format})
#  np.seterr(all="raise")
plt.rcParams["savefig.bbox"] = "tight"
# plt.rcParams["backend"] = "Agg"
plt.rcParams["figure.dpi"] = 150


def setup_ml():
    import torch

    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    random.seed(conf.seed)

    # Select the CPU/GPU
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
