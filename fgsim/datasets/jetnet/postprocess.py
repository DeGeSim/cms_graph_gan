from jetnet.utils import jet_features

from fgsim.config import conf
from fgsim.datasets.jetnet.utils import to_stacked_mask


def postprocess(batch):
    if len({"kpd", "fgd"} & set(conf.metrics.val)):
        from fgsim.datasets.jetnet.utils import to_efp

        batch = to_efp(batch)
    if "hlv" not in batch:
        batch["hlv"] = {}
    jn_dict = jet_features(to_stacked_mask(batch).cpu().numpy()[..., :3])
    for k, v in jn_dict.items():
        batch["hlv"][k] = v
    return batch
