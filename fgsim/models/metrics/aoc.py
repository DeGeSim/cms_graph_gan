import torch
from sklearn.metrics import roc_auc_score

from fgsim.config import conf, device


class Metric:
    def __init__(
        self,
    ):
        real_label = torch.ones(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )
        fake_label = torch.zeros(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )
        self.y_true = torch.hstack([real_label, fake_label]).detach().cpu().numpy()

    def __call__(self, d_gen: torch.Tensor, d_sim: torch.Tensor, **kwargs) -> float:
        sim_mean_disc = (
            d_sim.reshape(-1, conf.loader.batch_size).mean(dim=0).detach().cpu()
        )
        gen_mean_disc = (
            d_gen.reshape(-1, conf.loader.batch_size).mean(dim=0).detach().cpu()
        )
        y_pred = (
            torch.sigmoid(torch.hstack([sim_mean_disc, gen_mean_disc])).numpy()
            > 0.5
        )
        return float(roc_auc_score(self.y_true, y_pred))


aoc = Metric()
