# %%
# %load_ext autoreload
# %autoreload 2

import os

from omegaconf import OmegaConf

os.chdir(os.path.expanduser("~/fgsim"))

tagconf = OmegaConf.load("wd/jn_sched_test4/conf.yaml")
tagconf["tag"] = "jn_ac_dhlvs"
tagconf["command"] = "train"
tagconf["debug"] = True
import fgsim.config

fgsim.config.conf, _ = fgsim.config.compute_conf(fgsim.config.defaultconf, tagconf)


from fgsim.config import conf, device
from fgsim.ml.holder import Holder
from fgsim.ml.interactive_trainer import InteractiveTrainer
from fgsim.plot.model_plotter import model_plotter

model_plotter.active = True
# %%


class DeepTreePlotTrainer(InteractiveTrainer):
    def eval_step(self, res):
        fig = model_plotter.plot_model_outputs()
        # from IPython import display
        # display.clear_output(wait=True)
        # display.display(fig)
        # fig.show()
        fig.savefig("wd/modelplot.pdf")
        # from time import sleep
        # sleep(1)
        exit()


# %%
holder = Holder(device)
trainer = DeepTreePlotTrainer(holder)


trainer.training_loop()

# %%
