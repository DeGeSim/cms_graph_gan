from typing import Dict

import torch

from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder


class LossGen:
    # Ex∼pdata​(x)​[log(D(x))]+Ez∼pz​(z)​[log(1−D(G(z)))]
    # min for Gen, max​ for Disc

    def __init__(self, factor: float) -> None:
        self.factor = factor
        # sigmoid layer + Binary cross entropy
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def __call__(self, holder: Holder, batch: Batch) -> Dict[str, float]:
        # Loss of the simulated samples
        D_sim = holder.models.disc(batch)
        if isinstance(D_sim, dict):
            D_sim = torch.hstack(list(D_sim.values()))
        assert D_sim.dim() == 1
        # maximize log(D(x))
        # sample_disc_loss = -1 * torch.log(D_sim).mean() * self.factor
        # sample_disc_loss.backward()

        sample_disc_loss = self.criterion(D_sim, torch.ones_like(D_sim))
        sample_disc_loss.backward()

        # Loss of the generated samples
        # maximize log(1−D(G(z)))
        D_gen = holder.models.disc(holder.gen_points)
        if isinstance(D_gen, dict):
            D_gen = torch.hstack(list(D_gen.values()))
        assert D_gen.dim() == 1
        # gen_disc_loss = -1 * (
        #     torch.log(torch.ones_like(D_gen) - D_gen).mean() * self.factor
        # )
        # gen_disc_loss.backward()

        gen_disc_loss = self.criterion(D_gen, torch.zeros_like(D_gen))
        gen_disc_loss.backward()

        return {"gen": float(gen_disc_loss), "sim": float(sample_disc_loss)}
