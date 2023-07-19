import torch
from torch_geometric.data import Batch

from fgsim.config import conf


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(
        self,
        gen_batch: Batch,
        sim_batch: Batch,
        **kwargs,
    ):
        assert gen_batch.x.requires_grad
        assert not sim_batch.x.requires_grad

        losses = []

        for iftx in range(gen_batch.x.shape[-1]):
            simftx = sim_batch.x[:, iftx].sort()[0]
            genftx = gen_batch.x[:, iftx].sort()[0]
            s, g = scale_b_to_a(simftx, genftx)
            losses.append((s - g).abs().mean())

        epos = conf.loader.x_ftx_energy_pos

        for iftx in range(gen_batch.x.shape[-1]):
            if iftx == epos:
                continue
            simftx = (sim_batch.x[:, iftx] * sim_batch.x[:, epos]).sort()[0]
            genftx = (gen_batch.x[:, iftx] * gen_batch.x[:, epos]).sort()[0]
            s, g = scale_b_to_a(simftx, genftx)
            losses.append((s - g).abs().mean())

        return torch.stack(losses).mean()


def scale_b_to_a(a, b):
    assert not a.requires_grad
    mean, std = a.mean(), a.std()
    assert (std > 1e-6).all()
    sa = (a - mean) / (std + 1e-4)
    sb = (b - mean) / (std + 1e-4)
    return sa, sb
