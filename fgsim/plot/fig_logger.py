from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
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
    ) -> None:
        self.train_log = train_log
        self.plot_path = plot_path
        self.best_last_val = best_last_val
        self.step = step

    def __call__(self, figure: Figure, filename):
        figure.tight_layout()
        if hasattr(conf, "hash"):
            figure.text(
                0.05,
                0,
                f"#{conf.hash}",
                horizontalalignment="left",
                verticalalignment="bottom",
                fontsize=5,
            )
        figure.text(
            0.95,
            0,
            f"@{conf.tag}",
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=5,
        )
        figure.text(
            0.5,
            0,
            f"\nStep {self.step}",
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=5,
        )

        prefix = "/".join(self.best_last_val)

        if self.plot_path is not None:
            # figure.savefig((self.plot_path / filename
            # ).with_suffix(".png"), dpi=150)
            figure.savefig((self.plot_path / prefix / filename).with_suffix(".pdf"))

        self.train_log.log_figure(
            figure_name=f"{prefix}/{filename}",
            figure=figure,
            overwrite=False,
            step=self.step,
        )
        logger.info(filename)
        plt.close(figure)
