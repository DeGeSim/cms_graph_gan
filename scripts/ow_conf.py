# %%
from glob import glob
from os import chdir

import yaml

chdir("/home/mscham/fgsim")
# %%
files = glob("wd/cc_*/*/conf.yaml")
print(files)
# %%
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
