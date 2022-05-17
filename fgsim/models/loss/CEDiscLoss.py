from typing import Dict

import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from fgsim.config import conf, device
from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder


class LossGen:
    # Ex∼pdata​(x)​[log(D(x))]+Ez∼pz​(z)​[log(1−D(G(z)))]
    # min for Gen, max​ for Disc

    def __init__(self, factor: float) -> None:
        self.factor = factor
        # sigmoid layer + Binary cross entropy
        self.criterion = torch.nn.BCEWithLogitsLoss()
        real_label = torch.ones(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )
        fake_label = torch.zeros(
            (conf.loader.batch_size,), dtype=torch.float, device=device
        )
        self.y_true = torch.hstack([real_label, fake_label]).detach().cpu().numpy()

    def __call__(self, holder: Holder, batch: Batch) -> Dict[str, float]:
        # Loss of the simulated samples
        D_sim = holder.models.disc(batch)
        if isinstance(D_sim, dict):
            D_sim = torch.hstack(list(D_sim.values()))
        assert D_sim.dim() == 1
        # maximize log(D(x))
        # sample_disc_loss = -1 * torch.log(D_sim).mean() * self.factor
        # sample_disc_loss.backward()

        sample_disc_loss = self.criterion(D_sim, torch.ones_like(D_sim))
        sample_disc_loss.backward()

        # Loss of the generated samples
        # maximize log(1−D(G(z)))
        D_gen = holder.models.disc(holder.gen_points)
        if isinstance(D_gen, dict):
            D_gen = torch.hstack(list(D_gen.values()))
        assert D_gen.dim() == 1
        # gen_disc_loss = -1 * (
        #     torch.log(torch.ones_like(D_gen) - D_gen).mean() * self.factor
        # )
        # gen_disc_loss.backward()

        gen_disc_loss = self.criterion(D_gen, torch.zeros_like(D_gen))
        gen_disc_loss.backward()

        if not conf.debug and holder.state.grad_step % 10 == 0:
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
            cm = confusion_matrix(self.y_true, y_pred, normalize="true").ravel()
            for lossname, loss in zip(
                [
                    "true negative",
                    "false positive",
                    "false negative",
                    "true positive",
                ],
                cm,
            ):
                holder.train_log.log_loss(lossname, float(loss))

            holder.train_log.log_loss(
                "aoc", float(roc_auc_score(self.y_true, y_pred))
            )
            holder.train_log.log_loss(
                "average_precision_score",
                float(
                    average_precision_score(self.y_true, y_pred),
                ),
            )

        return {"gen": float(gen_disc_loss), "sim": float(sample_disc_loss)}
