from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(self, holder: Holder, *args, **kwargs):
        G_fake = holder.models.disc(holder.gen_points_w_grad)
        loss = G_fake.mean() * -1
        return loss
