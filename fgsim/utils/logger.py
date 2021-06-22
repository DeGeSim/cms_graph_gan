import logging

from ..config import conf

logger = logging.getLogger(__name__)

if not logger.handlers:
    format = "fgsim - %(levelname)s - %(message)s"

    logging.basicConfig(
        filename=conf.path.log,
        filemode="w",
        format=format,
    )
    logger.setLevel(logging.DEBUG if conf.debug else logging.INFO)

    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter(format)
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)
