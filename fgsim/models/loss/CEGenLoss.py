import torch

from fgsim.config import conf, device
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self, factor: float) -> None:
        self.factor = factor
        # sigmoid layer + Binary cross entropy
        self.criterion = torch.nn.BCEWithLogitsLoss()
        # self.fake_label = torch.zeros(
        #     (conf.loader.batch_size,), dtype=torch.float, device=device
        # )
        self.real_label = torch.ones(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )

    def __call__(self, holder: Holder, *args, **kwargs):
        # Loss of the generated samples

        D_gen = holder.models.disc(holder.gen_points_w_grad)
        if isinstance(D_gen, dict):
            D_gen = torch.hstack(list(D_gen.values()))
        assert D_gen.dim() == 1
        # minimize log(1−D(G(z)))
        # errG = (
        #     torch.log(torch.ones_like(D_gen) - D_gen).mean() * self.factor
        # )
        # instead
        # maximize log(D(G(z)))
        # decribed in
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        # errG = -1 * torch.log(D_gen).mean() * self.factor

        errG = self.criterion(D_gen, torch.ones_like(D_gen))
        errG.backward(retain_graph=True)

        return float(errG)
