import torch

from fgsim.config import conf, device
from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder


class LossGen:
    # Ex∼pdata​(x)​[log(D(x))]+Ez∼pz​(z)​[log(1−D(G(z)))]
    # min for Gen, max​ for Disc

    def __init__(self, factor: float) -> None:
        self.factor = factor
        self.criterion = torch.nn.BCELoss()
        self.real_label = torch.ones(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )
        self.fake_label = torch.zeros(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )

    def __call__(self, holder: Holder, batch: Batch) -> torch.float:
        # Loss of the simulated samples
        D_sim = holder.models.disc(batch)
        assert D_sim.dim() == 1
        # maximize log(D(x))
        # sample_disc_loss = -1 * torch.log(D_sim).mean() * self.factor
        # sample_disc_loss.backward()

        sample_disc_loss = self.criterion(D_sim, self.real_label)
        sample_disc_loss.backward()

        # Loss of the generated samples
        # maximize log(1−D(G(z)))
        D_gen = holder.models.disc(holder.gen_points)
        # gen_disc_loss = -1 * (
        #     torch.log(torch.ones_like(D_gen) - D_gen).mean() * self.factor
        # )
        # gen_disc_loss.backward()

        gen_disc_loss = self.criterion(D_gen, self.fake_label)
        gen_disc_loss.backward()

        return float(sample_disc_loss) + float(gen_disc_loss)
