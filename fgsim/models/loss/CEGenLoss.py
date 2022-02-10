import torch

from fgsim.config import conf, device
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, factor: float) -> None:
        self.factor = factor
        self.criterion = torch.nn.BCELoss()
        self.fake_label = torch.zeros(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )
        self.real_label = torch.ones(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )

    def __call__(self, holder: Holder, *args, **kwargs):
        # Loss of the generated samples

        D_gen = holder.models.disc(holder.gen_points_w_grad)
        # minimize log(1−D(G(z)))
        # errG = (
        #     torch.log(torch.ones_like(D_gen) - D_gen).mean() * self.factor
        # )
        # instead
        # maximize log(D(G(z)))
        # decribed in
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        # errG = -1 * torch.log(D_gen).mean() * self.factor

        errG = self.criterion(D_gen, self.real_label)
        errG.backward(retain_graph=True)

        return float(errG)
