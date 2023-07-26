# %%
from glob import glob

import yaml

# %%
files = glob("wd/cc*/*/conf.yaml")
for e in files:
    with open(e, "r") as f:
        cont = yaml.safe_load(f)
    cfg = cont["training"]["val"]["metrics"]
    if "cyratio" not in cfg:
        cfg.append("cyratio")
    if "implant_checkpoint" not in cont["training"]:
        cont["training"]["implant_checkpoint"] = False
    with open(e, "w") as f:
        yaml.safe_dump(cont, f)

# %%
