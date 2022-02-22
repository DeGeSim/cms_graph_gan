from dataclasses import dataclass
from typing import Dict

from fgsim.io.sel_seq import Batch
from fgsim.ml.holder import Holder


@dataclass
class LossGen:
    factor: float

    def __call__(self, holder: Holder, batch: Batch) -> Dict[str, float]:
        # EM dist loss:
        D_realm = holder.models.disc(batch).mean()
        sample_disc_loss = -D_realm * self.factor
        sample_disc_loss.backward()

        D_fakem = holder.models.disc(holder.gen_points).mean()
        gen_disc_loss = D_fakem * self.factor
        gen_disc_loss.backward()
        return {"gen": float(gen_disc_loss), "sim": float(sample_disc_loss)}
