import jetnet
import numpy as np
import torch
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.datasets.jetnet.utils import to_stacked_mask

num_batches = 1 if conf.command == "train" else 5
num_eval_samples = 50_000
jnkw = {"num_batches": num_batches, "num_eval_samples": num_eval_samples}


def w1m(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    jets1 = to_stacked_mask(gen_batch).detach().cpu().numpy()
    jets2 = to_stacked_mask(sim_batch).detach().cpu().numpy()
    jf1 = jets1[..., :3]
    jf2 = jets2[..., :3]
    score = jetnet.evaluation.gen_metrics.w1m(jets1=jf1, jets2=jf2, **jnkw)
    return tuple(min(float(e) * 1e3, 1e5) for e in score)


def w1p(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    jets1 = to_stacked_mask(gen_batch).detach().cpu().numpy()
    jets2 = to_stacked_mask(sim_batch).detach().cpu().numpy()
    jf1, jf2 = (jets1[..., :3], jets2[..., :3])
    mask1, mask2 = (jets1[..., -1], jets2[..., -1])

    score = jetnet.evaluation.gen_metrics.w1p(
        jets1=jf1, jets2=jf2, mask1=mask1, mask2=mask2, exclude_zeros=True, **jnkw
    )
    #  array of length num_particle_features
    # containing average W1 scores for each feature.
    # ["etarel", "phirel", "ptrel", "mask"]
    return (
        min(float(score[0].mean()) * 1e3, 1e5),
        min(float(np.sqrt((score[1] ** 2).sum()) * 1e3), 1e5),
    )


def w1efp(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    jets1 = to_stacked_mask(gen_batch).detach().cpu().numpy()
    jets2 = to_stacked_mask(sim_batch).detach().cpu().numpy()
    jf1, jf2 = (jets1[..., :3], jets2[..., :3])
    score = jetnet.evaluation.gen_metrics.w1efp(
        jets1=jf1, jets2=jf2, efp_jobs=10, **jnkw
    )
    return tuple(min(float(np.mean(e)) * 1e5, 1e5) for e in score)


def fpnd(gen_batch: Batch, **kwargs) -> float:
    jets = to_stacked_mask(gen_batch)[:, :, :3]
    if conf.loader.n_points != 30:
        pts = jets[..., conf.loader.x_ftx_energy_pos]
        topidxs = pts.topk(
            k=30,
            dim=1,
            largest=True,
            # sorted=True,
        ).indices

        highptjets = torch.stack([jet[idx] for jet, idx in zip(jets, topidxs)])

    else:
        highptjets = jets

    try:
        score = jetnet.evaluation.gen_metrics.fpnd(
            jets=highptjets,
            jet_type=conf.loader.jettype,
            use_tqdm=False,
        )
        return (min(float(score), 1e5), np.nan)
    except ValueError:
        return (1e5, np.nan)


def cov_mmd(gen_batch: Batch, sim_batch: Batch, **kwargs):
    try:
        real_jets = to_stacked_mask(sim_batch)
        gen_jets = to_stacked_mask(gen_batch)
        if gen_jets.isnan().any():
            raise ValueError
        score_cov, score_mmd = jetnet.evaluation.gen_metrics.cov_mmd(
            real_jets=real_jets,
            gen_jets=gen_jets,
            use_tqdm=False,
            num_batches=num_batches,
        )
        return min(float(score_cov), 1e5), min(float(score_mmd), 1e5)
    except ValueError:
        return (1e5, 1e5)


def kpd(
    sim_efps: torch.Tensor, gen_efps: torch.Tensor, **kwargs
) -> tuple[float, float]:
    score = jetnet.evaluation.gen_metrics.kpd(
        real_features=sim_efps,
        gen_features=gen_efps,
        num_threads=10,
        num_batches=num_batches,
    )
    return tuple(min(float(e) * 1e3, 1e5) for e in score)


def fpd(
    sim_efps: torch.Tensor, gen_efps: torch.Tensor, **kwargs
) -> tuple[float, float]:
    score = jetnet.evaluation.gen_metrics.fpd(
        real_features=sim_efps,
        gen_features=gen_efps,
        num_batches=num_batches,
    )
    return tuple(min(float(e) * 1e3, 1e5) for e in score)
