import torch
from sklearn.metrics import roc_auc_score


class Metric:
    def __init__(
        self,
    ):
        pass

    def __call__(self, d_gen: torch.Tensor, d_sim: torch.Tensor, **kwargs) -> float:
        assert d_sim.shape == d_gen.shape
        assert d_sim.dim() == 2 and d_sim.shape[1] == 1
        sim_mean_disc = d_sim.detach().cpu().reshape(-1)[:2000]
        gen_mean_disc = d_gen.detach().cpu().reshape(-1)[:2000]

        y_pred = torch.sigmoid(torch.hstack([sim_mean_disc, gen_mean_disc])).numpy()
        y_true = torch.hstack(
            [torch.ones_like(sim_mean_disc), torch.zeros_like(gen_mean_disc)]
        ).numpy()
        return float(roc_auc_score(y_true, y_pred))


auc = Metric()
