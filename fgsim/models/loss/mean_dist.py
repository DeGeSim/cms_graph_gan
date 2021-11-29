from dataclasses import dataclass
from typing import Optional

import torch

from fgsim.config import conf, device
from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    factor: float

    def __call__(self, holder: Holder, batch: Optional[torch.Tensor]):
        batch_means = torch.mean(batch, (0, 1))
        z = torch.randn(conf.loader.batch_size, 1, 96).to(device)
        tree = [z]
        fake_point = holder.models.gen(tree)
        fake_means = torch.mean(fake_point, (0, 1))
        loss = torch.F.mse_loss(fake_means, batch_means)
        return self.factor * loss
