from torchinfo import summary

from fgsim.monitoring.logger import logger


def log_model(holder):
    for partname, model in holder.models.parts.items():
        try:
            logger.info(f"Model {partname} Summary")
            logger.info(summary(model))
        except Exception:
            pass
