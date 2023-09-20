import torch

from fgsim.config import conf


def add_position(batch):
    from .objcol import scaler

    if batch.pos is not None:
        if len(batch.x) == len(batch.pos):
            return
    pos_l = []
    for iftx in range(batch.x.shape[1]):
        if iftx == conf.loader.x_ftx_energy_pos:
            continue
        pos_l.append(
            torch.tensor(
                scaler.transfs_x[iftx].inverse_transform(
                    batch.x[:, [iftx]].detach().cpu().double().numpy()
                )
            ).to(batch.x.device)
        )
    batch.pos = torch.hstack(pos_l).long()
