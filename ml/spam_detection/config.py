import pathlib
import sys

VOLUME_DIR = "/cache"
MODEL_STORE_DIR = pathlib.Path(VOLUME_DIR, "models")


def _get_logger():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, colorize=True)
    return logger
