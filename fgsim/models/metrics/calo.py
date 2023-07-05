from torch_geometric.data import Batch

from fgsim.config import conf

from .distutils import calc_cdf_dist, calc_wcdf_dist, scaled_w1
from .pca import fpc_from_batch
from .shower import analyze_shower, cone_ratio


# global variables
def w1cr(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    return scaled_w1(
        cone_ratio(sim_batch).reshape(-1), cone_ratio(gen_batch).reshape(-1)
    )


def w1fpc(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    sim_fpc = fpc_from_batch(sim_batch)
    gen_fpc = fpc_from_batch(gen_batch)
    w1s = [scaled_w1(s, g) for s, g in zip(sim_fpc.T, gen_fpc.T)]
    return dict(zip(["x", "y", "z"], w1s))


def w1z(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    sim_psr = analyze_shower(sim_batch)
    gen_psr = analyze_shower(gen_batch)

    w1s = {k: scaled_w1(sim_psr[k], gen_psr[k]) for k in sim_psr}
    return w1s


# Marginal Variables


def cdfdist(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    # Eralphz
    cdfdist = calc_cdf_dist(sim_batch.x, gen_batch.x)
    return dict(zip(conf.loader.x_features, cdfdist.detach().cpu().numpy()))


def cdfdist_Ew(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    # Eralphz
    cdfdist = calc_wcdf_dist(
        sim_batch.x[..., 1:],
        gen_batch.x[..., 1:],
        sim_batch.x[..., 0],
        gen_batch.x[..., 0],
    )
    return dict(zip(conf.loader.x_features[1:], cdfdist.detach().cpu().numpy()))


def w1mar(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    w1s = {
        k: scaled_w1(s, g)
        for s, g, k in zip(sim_batch.x.T, gen_batch.x.T, conf.loader.x_features)
    }
    return w1s
