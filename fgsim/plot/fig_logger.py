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
        if self.plot_path is not None:
            figure.savefig((self.plot_path / filename).with_suffix(".png"), dpi=150)
        label = conf.hash if hasattr(conf, "hash") else conf.tag
        figure.text(0, 0, label, fontsize=10)
        self.train_log.log_figure(
            figure_name=f"{self.best_last_val}/{filename}",
            figure=figure,
            overwrite=False,
            step=self.step,
        )
        logger.info(filename)
        plt.close(figure)
