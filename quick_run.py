# %%
# %load_ext autoreload
# %autoreload 2

from omegaconf import OmegaConf

tagconf = OmegaConf.load("wd/jn_ac_dhlvs/conf.yaml")
tagconf["tag"] = "jn_ac_dhlvs"
tagconf["command"] = "plot_model"
tagconf["debug"] = True
import fgsim.config

fgsim.config.conf, _ = fgsim.config.compute_conf(fgsim.config.defaultconf, tagconf)


from fgsim.config import conf, device
from fgsim.ml.holder import Holder
from fgsim.ml.interactive_trainer import InteractiveTrainer
from fgsim.plot.model_plotter import model_plotter

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
