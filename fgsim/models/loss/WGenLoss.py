from dataclasses import dataclass
from typing import Optional

import torch

from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    factor: float

    def __call__(self, holder: Holder, batch: Optional[torch.Tensor]):
        G_fake = holder.models.disc(holder.gen_points_w_grad)
        return self.factor * G_fake.mean() * -1
