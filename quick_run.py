# %%
# %load_ext autoreload
# %autoreload 2

from omegaconf import OmegaConf

tagconf = OmegaConf.load("wd/jnnoac/conf.yaml")
tagconf["tag"] = "jnnoac"
tagconf["command"] = "train"
tagconf["debug"] = True
import fgsim.config

fgsim.config.conf, _ = fgsim.config.compute_conf(fgsim.config.defaultconf, tagconf)
from fgsim.config import conf, device
from fgsim.ml.holder import Holder
from fgsim.ml.interactive_trainer import InteractiveTrainer

# %%
# %matplotlib inline
holder = Holder(device)
trainer = InteractiveTrainer(holder)
trainer.training_loop()


# %%
