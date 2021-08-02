import logging

from rich.highlighter import NullHighlighter
from rich.logging import RichHandler
from tqdm.contrib.logging import logging_redirect_tqdm

from ..config import conf

logger = logging.getLogger(__name__)

if not logger.handlers:
    format = "%(asctime)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        filename=conf.path.log,
        filemode="w",
        format=format,
        datefmt="%y-%m-%d %H:%M",
    )
    logger.setLevel(logging.DEBUG if conf.debug else logging.INFO)

    streamhandler = RichHandler(
        log_time_format="%y-%m-%d %H:%M", highlighter=NullHighlighter()
    )
    logger.addHandler(streamhandler)
    logging_redirect_tqdm(logger)
