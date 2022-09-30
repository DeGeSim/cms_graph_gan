"""This modules manages all objects that need to be available for the training:
Subnetworks, losses and optimizers. The Subnetworks and losses are dynamically imported,
depending on the config. Contains the code for checkpointing of model and optimzer status."""


import os
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from pathlib import Path

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from fgsim.config import conf
from fgsim.io.sel_loader import Batch
from fgsim.ml.loss import LossesCol
from fgsim.ml.network import SubNetworkCollector
from fgsim.ml.optim import OptimCol
from fgsim.ml.val_metrics import ValidationMetrics
from fgsim.monitoring.logger import logger
from fgsim.monitoring.train_log import TrainLog
from fgsim.utils.check_for_nans import contains_nans
from fgsim.utils.push_to_old import push_to_old


class Holder:
    """ "This class holds the models, the loss functions and the optimizers.
    It manages the checkpointing and holds a member 'state' that contains
    information about the current state of the training"""

    # Nameing convention snw = snw
    def __init__(self, device=torch.device("cpu")) -> None:
        self.device = device
        # Human readable, few values
        self.state: DictConfig = OmegaConf.create(
            {
                "epoch": 0,
                "processed_events": 0,
                "grad_step": 0,
                "complete": False,
            }
        )
        self.history = {
            "losses": {snwname: {} for snwname in conf.models},
            "val_metrics": {},
        }
        self.train_log = TrainLog(self.state, self.history)

        self.models: SubNetworkCollector = SubNetworkCollector(conf.models)
        self.models = self.models.float()
        # from torchinfo import summary
        # for _, model in self.models.parts.items():
        #     try:
        #         summary(model)
        #     except Exception:
        #         pass
        self.train_log.log_model_graph(self.models)
        self.losses: LossesCol = LossesCol(self.train_log)
        self.val_loss: ValidationMetrics = ValidationMetrics(self.train_log)
        self.optims: OptimCol = OptimCol(conf.models, self.models.get_par_dict())

        # try to load a check point
        self.checkpoint_loaded = False
        if (conf.command == "test" or not conf.debug) and not conf.ray:
            self.load_checkpoint()
        if conf.command == "test" and conf.ray:
            self.load_ray_checkpoint(
                sorted(glob(f"{conf.path.run_path}/checkpoint_*"))[-1]
            )

        # # Hack to move the optim parameters to the correct device
        # # https://github.com/pytorch/pytorch/issues/8741
        # with gpu_mem_monitor("optims"):
        #     self.optims.load_state_dict(self.optims.state_dict())

        logger.warning(f"Starting with state {str(OmegaConf.to_yaml(self.state))}")

        # Keep the generated samples ready, to be accessed by the losses
        self.gen_points: Batch = None
        self.gen_points_w_grad: Batch = None

        self._last_checkpoint_time = datetime.now()
        self._training_start_time = datetime.now()
        self.saved_first_checkpoint = False

        # checking
        # import torcheck

        # for partname, model in self.models.parts.items():
        #     torcheck.register(self.optims[partname])
        #     torcheck.add_module_changing_check(model, module_name=partname)
        #     # torcheck.add_module_inf_check(model, module_name=partname)
        #     # torcheck.add_module_nan_check(model, module_name=partname)
        self.to(self.device)

    def to(self, device):
        self.device = device
        self.models = self.models.to(device)
        self.optims.to(device)

        return self

    def load_checkpoint(self):
        if not (
            os.path.isfile(conf.path.state) and os.path.isfile(conf.path.checkpoint)
        ):
            if conf.command != "train":
                raise FileNotFoundError("Could not find checkpoint")
            logger.warning("Proceeding without loading checkpoint.")
            return
        self.state = OmegaConf.load(conf.path.state)
        # Once the state has been loaded from the checkpoint,
        #  update the logger state
        self.train_log.state = self.state
        checkpoint = torch.load(conf.path.checkpoint, map_location=self.device)

        assert not contains_nans(checkpoint["models"])[0]
        assert not contains_nans(checkpoint["best_model"])[0]

        self.models.load_state_dict(checkpoint["models"])
        self.optims.load_state_dict(checkpoint["optims"])
        self.best_model_state = checkpoint["best_model"]
        self.history = checkpoint["history"]
        self.checkpoint_loaded = True

        logger.warning(
            "Loaded model from checkpoint at"
            + f" epoch {self.state['epoch']}"
            + f" grad_step {self.state['grad_step']}."
        )

    def load_ray_checkpoint(self, ray_tmp_checkpoint_path: str):
        checkpoint_path = Path(ray_tmp_checkpoint_path) / "cp.pth"
        state_path = Path(ray_tmp_checkpoint_path) / "state.pth"
        self.state = OmegaConf.load(state_path)
        # Once the state has been loaded from the checkpoint,
        #  update the logger state
        self.train_log.state = self.state
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        assert not contains_nans(checkpoint["models"])[0]
        assert not contains_nans(checkpoint["best_model"])[0]

        self.models.load_state_dict(checkpoint["models"])
        self.optims.load_state_dict(checkpoint["optims"])
        self.best_model_state = checkpoint["best_model"]
        self.history = checkpoint["history"]
        self.checkpoint_loaded = True

        logger.warning(
            f"Loaded model from checkpoint at epoch {self.state['epoch']}"
            + f" grad_step {self.state['grad_step']} from {ray_tmp_checkpoint_path}"
        )

    def select_best_model(self):
        if conf.ray and conf.command == "test":
            return
        self.models.load_state_dict(self.best_model_state)
        self.models = self.models.float().to(self.device)

    def save_checkpoint(
        self,
    ):
        if conf.debug:
            return
        push_to_old(conf.path.checkpoint, conf.path.checkpoint_old)
        torch.save(
            {
                "models": self.models.state_dict(),
                "optims": self.optims.state_dict(),
                "best_model": self.best_model_state,
                "history": self.history,
            },
            conf.path.checkpoint,
        )
        push_to_old(conf.path.state, conf.path.state_old)
        OmegaConf.save(config=self.state, f=conf.path.state)
        self._last_checkpoint_time = datetime.now()
        logger.warning(
            f"{self._last_checkpoint_time.strftime('%d/%m/%Y, %H:%M:%S')}"
            f"Checkpoint saved to {conf.path.checkpoint}"
        )

    def save_ray_checkpoint(self, ray_tmp_checkpoint_path: str):
        checkpoint_path = Path(ray_tmp_checkpoint_path) / "cp.pth"
        state_path = Path(ray_tmp_checkpoint_path) / "state.pth"
        torch.save(
            {
                "models": self.models.state_dict(),
                "optims": self.optims.state_dict(),
                "best_model": self.best_model_state,
                "history": self.history,
            },
            checkpoint_path,
        )
        OmegaConf.save(config=self.state, f=state_path)
        self._last_checkpoint_time = datetime.now()
        logger.warning(
            f"{self._last_checkpoint_time.strftime('%d/%m/%Y, %H:%M:%S')}"
            f"Checkpoint saved to {ray_tmp_checkpoint_path}"
        )
        return ray_tmp_checkpoint_path

    def checkpoint_after_time(self):
        now = datetime.now()
        time_since_last_checkpoint = (
            now - self._last_checkpoint_time
        ).seconds // 60
        interval = conf.training.checkpoint_minutes

        if time_since_last_checkpoint > interval:
            self.save_checkpoint()

        time_since_training_start = (now - self._training_start_time).seconds // 60
        if time_since_training_start > 5 and not self.saved_first_checkpoint:
            self.saved_first_checkpoint = True
            self.save_checkpoint()

    def pass_batch_through_model(
        self, sim_batch, train_gen: bool = False, train_disc: bool = False
    ):
        assert not (train_gen and train_disc)
        # generate the random vector
        z = torch.randn(*self.models.gen.z_shape, requires_grad=True).to(
            self.device
        )
        if len(conf.loader.y_features) == 0:
            cond = torch.empty(
                (conf.loader.batch_size, 0), dtype=torch.float, device=self.device
            )
        else:
            cond = sim_batch.y
        with with_grad(train_gen):
            gen_batch = self.models.gen(z, cond)

        # In both cases the gradient needs to pass though d_gen
        with with_grad(train_gen or train_disc):
            d_gen = self.models.disc(gen_batch, cond).squeeze()

        self.res = {
            "sim_batch": sim_batch,
            "gen_batch": gen_batch,
            "d_gen": d_gen,
        }
        # we dont need to compute d_sim if only the generator is trained
        # but we need it for the validation
        if train_disc or (train_disc == train_gen):
            with with_grad(train_disc):
                d_sim = self.models.disc(sim_batch, cond).squeeze()
            self.res["d_sim"] = d_sim
        return self.res


@contextmanager
def with_grad(condition):
    if not condition:
        with torch.no_grad():
            yield
    else:
        yield
