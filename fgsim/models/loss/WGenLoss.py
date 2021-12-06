from dataclasses import dataclass

from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    factor: float

    def __call__(self, holder: Holder, *args, **kwargs):
        G_fake = holder.models.disc(holder.gen_points_w_grad.pc)
        return self.factor * G_fake.mean() * -1
