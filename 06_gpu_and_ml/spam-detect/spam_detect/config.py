import enum
import pathlib
import sys

VOLUME_DIR = "/cache"
MODEL_STORE_DIR = pathlib.Path(VOLUME_DIR, "models")
MODEL_REGISTRY_FILENAME = "registry.json"
DATA_DIR = pathlib.Path(VOLUME_DIR, "data")


class ModelTypes(str, enum.Enum):
    BAD_WORDS = "BAD_WORDS"
    LLM = "LLM"
    NAIVE_BAYES = "NAIVE_BAYES"


def get_logger():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, colorize=True, level="WARNING")
    return logger
