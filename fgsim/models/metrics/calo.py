from torch_geometric.data import Batch

from fgsim.config import conf

from .distutils import calc_cdf_dist, calc_wcdf_dist, scaled_w1

# Marginal Variables


def marginal(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    # Eralphz
    cdfdist = calc_cdf_dist(sim_batch.x, gen_batch.x)
    return dict(zip(conf.loader.x_features, cdfdist.detach().cpu().numpy()))


def marginalEw(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    # Eralphz
    cdfdist = calc_wcdf_dist(
        sim_batch.x[..., 1:],
        gen_batch.x[..., 1:],
        sim_batch.x[..., 0],
        gen_batch.x[..., 0],
    )
    return dict(zip(conf.loader.x_features[1:], cdfdist.detach().cpu().numpy()))


# global variables
def coneratio(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    return scaled_w1(sim_batch.coneratio, gen_batch.coneratio)


def fpc(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    w1s = {
        k: scaled_w1(sim_batch["fpc"][k], gen_batch["fpc"][k])
        for k in ["x", "y", "z"]
    }
    return w1s


def showershape(gen_batch: Batch, sim_batch: Batch, **kwargs) -> dict[str, float]:
    sim_psr = sim_batch["showershape"]
    gen_psr = gen_batch["showershape"]

    w1s = {k: scaled_w1(sim_psr[k], gen_psr[k]) for k in sim_psr}
    return w1s
