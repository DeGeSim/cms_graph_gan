import torch
from scipy.stats import wasserstein_distance


class Metric:
    def __init__(
        self,
    ):
        pass

    def __call__(self, d_gen: torch.Tensor, d_sim: torch.Tensor, **kwargs) -> float:
        assert d_sim.shape == d_gen.shape
        assert d_sim.dim() == 2 and d_sim.shape[1] == 1
        return wasserstein_distance(
            d_gen.detach().cpu().numpy().reshape(-1)[:2000],
            d_sim.detach().cpu().numpy().reshape(-1)[:2000],
        )


w1disc = Metric()
