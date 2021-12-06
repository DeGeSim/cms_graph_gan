from dataclasses import dataclass

import torch

from fgsim.io.queued_dataset import Batch
from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    factor: float

    def __call__(self, holder: Holder, batch: Batch) -> torch.float:
        # EM dist loss:
        D_realm = holder.models.disc(batch.pc).mean()
        D_fakem = holder.models.disc(holder.gen_points.pc).mean()
        d_loss = -D_realm + D_fakem
        return self.factor * d_loss
