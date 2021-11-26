from dataclasses import dataclass

import torch

from fgsim.config import conf, device
from fgsim.io.queued_dataset import BatchType
from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    factor: float

    def __call__(self, holder: Holder, batch: BatchType) -> torch.float:
        # EM dist loss:
        D_realm = holder.models.disc(batch).mean()

        z = torch.randn(conf.loader.batch_size, 1, 96).to(device)
        tree = [z]
        with torch.no_grad():
            fake_point = holder.models.gen(tree)
        D_fakem = holder.models.disc(fake_point).mean()
        d_loss = -D_realm + D_fakem
        return self.factor * d_loss
