import logging

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

    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter(format, datefmt="%y-%m-%d %H:%M")
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
