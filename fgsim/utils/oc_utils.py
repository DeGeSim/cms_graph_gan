import hashlib
from typing import List

from omegaconf import DictConfig, OmegaConf


# Exclude the keys that do not affect the training
def removekeys(omconf: DictConfig, excluded_keys: List[str]) -> DictConfig:
    filtered_omconf = OmegaConf.masked_copy(
        omconf,
        [k for k in omconf.keys() if k not in excluded_keys],
    )
    return filtered_omconf


def dict_to_kv(o, keystr=""):
    """Converts a nested dict {"a":"foo", "b": {"foo":"bar"}} to \
    [("a","foo"),("b.foo","bar")]."""
    if hasattr(o, "keys"):
        outL = []
        for k in o.keys():
            elemres = dict_to_kv(o[k], keystr + str(k) + ".")
            if (
                len(elemres) == 2
                and type(elemres[0]) == str
                and type(elemres[1]) == str
            ):
                outL.append(elemres)
            else:
                for e in elemres:
                    outL.append(e)
        return outL
    elif hasattr(o, "__str__"):

        return (keystr.strip("."), str(o))
    else:
        raise ValueError


# convert the config to  key-value pairs
# sort them, hash the results
def gethash(omconf: DictConfig) -> str:
    OmegaConf.resolve(omconf)
    kv_list = [f"{e[0]}: {e[1]}" for e in dict_to_kv(omconf)]
    kv_str = "\n".join(sorted(kv_list))
    omhash = str(hashlib.sha1(kv_str.encode()).hexdigest()[:7])
    return omhash
