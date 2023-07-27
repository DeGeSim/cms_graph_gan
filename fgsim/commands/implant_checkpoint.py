"""
Given a trained model, it will generate a set of random events
and compare the generated events to the simulated events.
"""


from pathlib import Path

import torch

from fgsim.config import conf
from fgsim.ml.holder import Holder

holder = Holder()


checkpoint = torch.load(Path(conf.path.checkpoint))


def filter_dict(d, valid_keys):
    d = {k: v for k, v in d.items() if k in valid_keys}
    if "spectral_norm" in conf.models.disc.params.bipart_param:
        if conf.models.disc.params.bipart_param.spectral_norm:
            d = {
                k: v
                for k, v in d.items()
                if k
                not in [
                    "disc.pools.0.mpl.lin_l.weight",
                    "disc.pools.1.mpl.lin_l.weight",
                    "disc.pools.2.mpl.lin_l.weight",
                ]
            }
    assert set(d.keys()).issubset(set(valid_keys))
    return d


valid_keys_model = holder.models.state_dict().keys()

checkpoint["models"] = filter_dict(checkpoint["models"], valid_keys_model)
checkpoint["best_model"] = filter_dict(checkpoint["best_model"], valid_keys_model)


if len(holder.swa_models):
    for pname, part in holder.swa_models.items():
        valid_keys = part.state_dict().keys()
        checkpoint["swa_model"][pname] = filter_dict(
            checkpoint["swa_model"][pname], valid_keys
        )
        checkpoint["best_swa_model"][pname] = filter_dict(
            checkpoint["best_swa_model"][pname], valid_keys
        )


valid_keys_optim = holder.optims.state_dict().keys()
checkpoint["optims"] = filter_dict(checkpoint["optims"], valid_keys_optim)
torch.save(checkpoint, Path(conf.path.checkpoint))
