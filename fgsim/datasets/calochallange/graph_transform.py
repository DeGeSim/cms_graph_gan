import torch
from caloutils import calorimeter
from caloutils.processing import voxel_to_pc

from .convcoord import Exyz_to_Ezalphar, Ezalphar_to_Exyz
from .readin import read_chunks

# from fgsim.io.dequantscaler import dequant


__safety_gap = 1e-6


def dequant(x):
    noise = torch.rand(*x.shape, dtype=torch.double)
    xnew = x.double() + torch.clip(noise, __safety_gap, 1 - __safety_gap)
    assert (x == torch.floor(xnew)).all()
    return xnew


def events_to_batch(chks: tuple[torch.Tensor, torch.Tensor]):
    Es, showers = [torch.stack(e) for e in zip(*read_chunks(chks))]
    batch = voxel_to_pc(showers.reshape(-1, *calorimeter.dims), Es.squeeze())
    E, num_hits = batch.y.T
    batch["y"] = torch.stack([E, num_hits, E / num_hits], -1)

    for i in [1, 2, 3]:
        icoord = batch.x[:, i]
        icoord = dequant(icoord.int())
        icoord = (
            icoord
            / [
                None,
                calorimeter.num_z,
                calorimeter.num_alpha,
                calorimeter.num_r,
            ][i]
        )
        assert (icoord <= 1).all()
        batch.x[:, i] = icoord
    Exyz = Ezalphar_to_Exyz(batch.x)
    assert (batch.x - Exyz_to_Ezalphar(Exyz)).abs().max() < 1e-6
    batch.x = Exyz

    return batch
