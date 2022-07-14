from typing import Dict

from fgsim.io.sel_loader import Batch
from fgsim.ml.holder import Holder


class LossGen:
    def __init__(self) -> None:
        pass

    def __call__(self, holder: Holder, batch: Batch) -> Dict[str, float]:
        # EM dist loss:
        D_realm = holder.models.disc(batch).mean()
        sample_disc_loss = -D_realm

        D_fakem = holder.models.disc(holder.gen_points).mean()
        gen_disc_loss = D_fakem
        return gen_disc_loss + sample_disc_loss
