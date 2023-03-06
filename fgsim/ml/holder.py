"""This modules manages all objects that need to be available for the training:
Subnetworks, losses and optimizers. The Subnetworks and losses are dynamically imported,
depending on the config. Contains the code for checkpointing of model and optimzer status."""


import os
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from glob import glob
from pathlib import Path

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch.optim.swa_utils import AveragedModel

from fgsim.config import conf
from fgsim.io.sel_loader import Batch
from fgsim.ml.loss import LossesCol
from fgsim.ml.network import SubNetworkCollector
from fgsim.ml.optim import OptimAndSchedulerCol
from fgsim.ml.val_metrics import ValidationMetrics
from fgsim.monitoring import TrainLog, logger
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
            "losses": {snwname: defaultdict(list) for snwname in conf.models},
            "val": defaultdict(list),
        }
        self.train_log = TrainLog(self.state, self.history)

        self.models: SubNetworkCollector = SubNetworkCollector(conf.models)
        # For SWA the models need to be on the right device before initializing
        self.models = self.models.float().to(device)
        self.train_log.log_model_graph(self.models)
        self.swa_models = {
            k: AveragedModel(v) for k, v in self.models.parts.items()
        }

        if conf.command == "train":
            self.optims: OptimAndSchedulerCol = OptimAndSchedulerCol(
                conf.models, self.models, self.swa_models, self.train_log
            )

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

        # import torcheck
        # for partname, model in self.models.parts.items():
        #     torcheck.register(self.optims[partname])
        #     torcheck.add_module_changing_check(model, module_name=partname)
        #     # torcheck.add_module_inf_check(model, module_name=partname)
        #     # torcheck.add_module_nan_check(model, module_name=partname)

        self.losses: LossesCol = LossesCol(self.train_log)
        self.val_metrics: ValidationMetrics = ValidationMetrics(self.train_log)

        self.to(self.device)

    def to(self, device):
        self.device = device
        self.models = self.models
        # self.swa_models = {k: v.to(device) for k, v in self.swa_models.items()}
        # self.swa_models = {
        #     k: AveragedModel(v).to(device) for k, v in self.models.parts.items()
        # }
        if conf.command == "train":
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
        self._load_checkpoint_path(
            Path(conf.path.state), Path(conf.path.checkpoint)
        )

    def load_ray_checkpoint(self, ray_tmp_checkpoint_path: str):
        checkpoint_path = Path(ray_tmp_checkpoint_path) / "cp.pth"
        state_path = Path(ray_tmp_checkpoint_path) / "state.pth"
        self._load_checkpoint_path(state_path, checkpoint_path)

    def _load_checkpoint_path(self, state_path, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        assert not contains_nans(checkpoint["models"])[0]
        assert not contains_nans(checkpoint["best_model"])[0]

        self.models.load_state_dict(checkpoint["models"])
        if conf.command == "train":
            self.optims.load_state_dict(checkpoint["optims"])
        self.best_model_state = checkpoint["best_model"]

        self.history.update(checkpoint["history"])
        self.state.update(OmegaConf.load(state_path))
        self.checkpoint_loaded = True

        logger.warning(
            "Loaded model from checkpoint at"
            + f" epoch {self.state['epoch']}"
            + f" grad_step {self.state['grad_step']}."
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
                "optims": cylerlr_workaround(self.optims.state_dict()),
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
                "optims": cylerlr_workaround(self.optims.state_dict()),
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
        self,
        sim_batch,
        train_gen: bool = False,
        train_disc: bool = False,
        eval=False,
    ):
        assert not (train_gen and train_disc)
        assert not (eval and (train_gen or train_disc))
        assert not torch.isnan(sim_batch.x).any()
        assert sim_batch.y.shape[-1] == len(conf.loader.y_features)
        assert sim_batch.x.shape[-1] == len(conf.loader.x_features)
        # if eval:
        #     self.models.eval()
        # else:
        #     self.models.train()

        if eval and conf["models"]["gen"]["scheduler"]["name"] == "SWA":
            gen = self.swa_models["gen"]
        else:
            gen = self.models.gen

        if eval and conf["models"]["disc"]["scheduler"]["name"] == "SWA":
            disc = self.swa_models["disc"]
        else:
            disc = self.models.disc

        # generate the random vector
        z = torch.randn(
            *self.models.gen.z_shape,
            requires_grad=True,
            dtype=torch.float,
            device=self.device,
        )
        if len(conf.loader.y_features) == 0:
            cond = torch.empty(
                (conf.loader.batch_size, 0), dtype=torch.float, device=self.device
            )
        else:
            cond = sim_batch.y

        with with_grad(train_gen):
            gen_batch = gen(z, cond)
        assert not torch.isnan(gen_batch.x).any()
        assert sim_batch.x.shape[-1] == gen_batch.x.shape[-1]

        # In both cases the gradient needs to pass though d_gen
        with with_grad(train_gen or train_disc):
            d_gen_out = disc(gen_batch, cond)
        # Save the latent features for the feature matching loss
        if isinstance(d_gen_out, tuple):
            d_gen, d_gen_latftx = d_gen_out
        else:
            d_gen, d_gen_latftx = d_gen_out, None

        assert d_gen.shape == (conf.loader.batch_size, 1)
        assert not torch.isnan(d_gen).any()

        res = {
            "sim_batch": sim_batch,
            "gen_batch": gen_batch,
            "d_gen": d_gen,
            "d_gen_latftx": d_gen_latftx,
        }
        # we dont need to compute d_sim if only the generator is trained
        # but we need it for the validation
        # and for the feature matching loss
        if (
            train_disc
            or (train_disc == train_gen)
            or ("feature_matching" in conf.models.gen.losses and train_gen)
        ):
            with with_grad(train_disc):
                d_sim_out = disc(sim_batch, cond)
            # Save the latent features for the feature matching loss
            if isinstance(d_sim_out, tuple):
                d_sim, d_sim_latftx = d_sim_out
            else:
                d_sim, d_sim_latftx = d_sim_out, None
            assert d_sim.shape == (conf.loader.batch_size, 1)
            res["d_sim"] = d_sim
            res["d_sim_latftx"] = d_sim_latftx
        return res


@contextmanager
def with_grad(condition):
    if not condition:
        with torch.no_grad():
            yield
    else:
        yield


def cylerlr_workaround(sd):
    for pname in sd["schedulers"]:
        if "_scale_fn_ref" in sd["schedulers"][pname]:
            del sd["schedulers"][pname]["_scale_fn_ref"]
    return sd
