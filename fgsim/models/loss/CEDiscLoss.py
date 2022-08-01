import torch

from fgsim.io.sel_loader import Batch
from fgsim.ml.holder import Holder


class LossGen:
    # Ex∼pdata​(x)​[log(D(x))]+Ez∼pz​(z)​[log(1−D(G(z)))]
    # min for Gen, max​ for Disc

    def __init__(self) -> None:
        # sigmoid layer + Binary cross entropy
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def __call__(self, holder: Holder, batch: Batch) -> torch.Tensor:
        # Loss of the simulated samples
        D_sim = holder.models.disc(batch).squeeze()
        if isinstance(D_sim, dict):
            D_sim = torch.hstack(list(D_sim.values()))
        assert D_sim.dim() == 1
        # maximize log(D(x))
        # sample_disc_loss = -1 * torch.log(D_sim).mean()

        sample_disc_loss = self.criterion(D_sim, torch.ones_like(D_sim))

        # Loss of the generated samples
        # maximize log(1−D(G(z)))
        D_gen = holder.models.disc(holder.gen_points).squeeze()
        if isinstance(D_gen, dict):
            D_gen = torch.hstack(list(D_gen.values()))
        assert D_gen.dim() == 1
        # gen_disc_loss = -1 * (
        #     torch.log(torch.ones_like(D_gen) - D_gen).mean()
        # )

        gen_disc_loss = self.criterion(D_gen, torch.zeros_like(D_gen))

        return gen_disc_loss + sample_disc_loss
