import torch
from caloutils import calorimeter
from caloutils.processing import voxel_to_pc

from .readin import read_chunks


def events_to_batch(chks: tuple[torch.Tensor, torch.Tensor]):
    Es, showers = [torch.stack(e) for e in zip(*read_chunks(chks))]
    batch = voxel_to_pc(showers.reshape(-1, *calorimeter.dims), Es.squeeze())
    E, num_hits = batch.y.T
    batch["y"] = torch.stack([E, num_hits, E / num_hits], -1)
    return batch
