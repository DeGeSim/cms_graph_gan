from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from fgsim.config import conf
from fgsim.monitoring import TrainLog, logger


class FigLogger:
    def __init__(
        self,
        train_log: TrainLog,
        plot_path: Optional[Path],
        best_last_val: list[str],
        step: int,
        epoch: int,
    ) -> None:
        self.train_log = train_log
        self.plot_path = plot_path
        self.best_last_val = best_last_val
        self.step = step
        self.epoch = epoch

    def __call__(self, figure: Figure, filename):
        figure.tight_layout()
        texts = [
            f"@{conf.tag}",
            f"Step {self.step}",
            f"epoch {self.epoch}",
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ]
        if hasattr(conf, "hash"):
            texts.append(f"#{conf.hash}")
        xposv = np.linspace(0.05, 0.95, len(texts))
        for x, t in zip(xposv, texts):
            figure.text(
                x,
                0,
                t,
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=5,
            )

        prefix = "/".join(self.best_last_val)

        if self.plot_path is not None:
            # figure.savefig((self.plot_path / filename
            # ).with_suffix(".png"), dpi=150)
            folder = (self.plot_path / prefix).mkdir(parents=True, exist_ok=True)
            figure.savefig((folder / filename).with_suffix(".pdf"))

        self.train_log.log_figure(
            figure_name=f"{prefix}/{filename}",
            figure=figure,
            overwrite=False,
            step=self.step,
        )
        logger.info(filename)
        plt.close(figure)
