import logging

import queueflow as qf
from multiprocessing_logging import install_mp_handler
from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from tqdm.contrib.logging import logging_redirect_tqdm

from fgsim.config import conf

logger = logging.getLogger(__name__)


def init_logger():
    if not logger.handlers:
        log_path = f"{conf.path.run_path}/{conf.command}.log"
        loader_log = f"{conf.path.run_path}/{conf.command}_loader.log"
        if not conf.debug:
            logging.basicConfig(
                filename=log_path,
                level=logging.DEBUG if conf.loader.debug else logging.INFO,
                filemode="w",
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%y-%m-%d %H:%M",
            )
            qf.logger.setup_logger(
                loader_log,
                print_bool=conf.loader.debug,
                debug=conf.loader.debug,
            )

        streamhandler = RichHandler(
            log_time_format="%y-%m-%d %H:%M", highlighter=NullHighlighter()
        )
        logger.addHandler(streamhandler)

        logger.setLevel(logging.DEBUG if conf.loader.debug else logging.INFO)
        logging_redirect_tqdm([logger])
        install_mp_handler(logger)
