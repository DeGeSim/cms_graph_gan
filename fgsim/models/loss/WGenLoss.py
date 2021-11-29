from dataclasses import dataclass
from typing import Optional

import torch

from fgsim.config import conf, device
from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    factor: float

    def __call__(self, holder: Holder, batch: Optional[torch.Tensor]):
        z = torch.randn(conf.loader.batch_size, 1, 96).to(device)
        fake_point = holder.models.gen(z)
        G_fake = holder.models.disc(fake_point)
        return self.factor * G_fake.mean() * -1
