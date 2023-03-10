import jetnet
from torch_geometric.data import Batch

from fgsim.config import conf
from fgsim.utils.jetnetutils import to_stacked_mask

jet_type = conf.loader.jettype


def w1m(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    score = jetnet.evaluation.gen_metrics.w1m(
        jets1=to_stacked_mask(gen_batch)[..., :3],
        jets2=to_stacked_mask(sim_batch)[..., :3],
    )
    return tuple(min(float(e) * 1e3, 1e5) for e in score)


def w1p(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    jets1 = to_stacked_mask(gen_batch).detach().cpu().numpy()
    jets2 = to_stacked_mask(sim_batch).detach().cpu().numpy()
    score = jetnet.evaluation.gen_metrics.w1p(
        jets1=jets1[..., :3], jets2=jets2[..., :3], exclude_zeros=True
    )
    return tuple(min(float(e) * 1e3, 1e5) for e in score)


def w1efp(gen_batch: Batch, sim_batch: Batch, **kwargs) -> tuple[float, float]:
    score = jetnet.evaluation.gen_metrics.w1efp(
        jets1=to_stacked_mask(gen_batch)[..., :3].cpu(),
        jets2=to_stacked_mask(sim_batch)[..., :3].cpu(),
        efp_jobs=12,
    )
    return tuple(min(float(e) * 1e5, 1e5) for e in score)


def fpnd(gen_batch: Batch, **kwargs) -> float:
    try:
        score = jetnet.evaluation.gen_metrics.fpnd(
            jets=to_stacked_mask(gen_batch)[..., :3],
            jet_type=jet_type,
            use_tqdm=False,
        )
        return min(float(score), 1e5)
    except ValueError:
        return 1e5


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
        )
        return min(float(score_cov), 1e5), min(float(score_mmd), 1e5)
    except ValueError:
        return (1e5, 1e5)
