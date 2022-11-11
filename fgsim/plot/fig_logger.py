import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from fgsim.config import conf
from fgsim.monitoring.logger import logger


class FigLogger:
    def __init__(self, train_log, plot_path, best_last_val, step) -> None:
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
                fontsize=6,
            )
        figure.text(
            0.95,
            0,
            f"@{conf.tag}",
            horizontalalignment="right",
            verticalalignment="bottom",
            fontsize=6,
        )

        if self.plot_path is not None:
            figure.savefig((self.plot_path / filename).with_suffix(".png"), dpi=150)

        self.train_log.log_figure(
            figure_name=f"{self.best_last_val}/{filename}",
            figure=figure,
            overwrite=False,
            step=self.step,
        )
        logger.info(filename)
        plt.close(figure)
