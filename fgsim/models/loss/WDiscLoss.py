from dataclasses import dataclass

import torch

from fgsim.io.queued_dataset import BatchType
from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    factor: float

    def __call__(self, holder: Holder, batch: BatchType) -> torch.float:
        # EM dist loss:
        D_realm = holder.models.disc(batch).mean()
        D_fakem = holder.models.disc(holder.gen_points).mean()
        d_loss = -D_realm + D_fakem
        return self.factor * d_loss
