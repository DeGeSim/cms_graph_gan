import torch

from fgsim.io.sel_loader import Batch
from fgsim.ml.holder import Holder


class LossGen:
    # Ex∼pdata​(x)​[log(D(x))]+Ez∼pz​(z)​[log(1−D(G(z)))]
    # min for Gen, max​ for Disc

    def __init__(self) -> None:
        # sigmoid layer + Binary cross entropy
        self.lossf = torch.nn.MSELoss()

    def __call__(self, holder: Holder, batch: Batch) -> torch.Tensor:
        # Loss of the simulated samples
        D_sim = holder.models.disc(batch)
        D_gen = holder.models.disc(holder.gen_points)
        return self.lossf(D_gen, D_sim)
