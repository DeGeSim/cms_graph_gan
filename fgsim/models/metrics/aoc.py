import torch
from sklearn.metrics import roc_auc_score

from fgsim.config import conf, device
from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder


class LossGen:
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

    def __call__(self, holder: Holder, batch: Batch) -> float:

        D_sim = holder.models.disc(batch)
        D_gen = holder.models.disc(holder.gen_points)
        sim_mean_disc = (
            D_sim.reshape(-1, conf.loader.batch_size).mean(dim=0).detach().cpu()
        )
        gen_mean_disc = (
            D_gen.reshape(-1, conf.loader.batch_size).mean(dim=0).detach().cpu()
        )
        y_pred = (
            torch.sigmoid(torch.hstack([sim_mean_disc, gen_mean_disc])).numpy()
            > 0.5
        )

        # cm = confusion_matrix(self.y_true, y_pred, normalize="true").ravel()
        # for lossname, loss in zip(
        #     [
        #         "true negative",
        #         "false positive",
        #         "false negative",
        #         "true positive",
        #     ],
        #     cm,
        # ):
        #     holder.train_log.log_loss(lossname, float(loss))

        # holder.train_log.log_loss(
        #     "average_precision_score",
        #     float(
        #         average_precision_score(self.y_true, y_pred),
        #     ),
        # )

        return float(roc_auc_score(self.y_true, y_pred))
