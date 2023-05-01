import enum
import logging
import pathlib
import sys

VOLUME_DIR: str = "/cache"
MODEL_STORE_DIR = pathlib.Path(VOLUME_DIR, "models")
MODEL_REGISTRY_FILENAME: str = "registry.json"
DATA_DIR = pathlib.Path(VOLUME_DIR, "data")

SERVING_MODEL_ID: str = (
    "sha256.4D4CA273952449C9D20E837F4425DC012C1BABF9AFD4D8E118BB50A596C72B87"
)


class ModelType(str, enum.Enum):
    BAD_WORDS = "BAD_WORDS"
    LLM = "LLM"
    NAIVE_BAYES = "NAIVE_BAYES"


def get_logger():
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, colorize=True, level="INFO")
    except ModuleNotFoundError:
        logger = logging.getLogger()
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(levelname)s: %(asctime)s: %(name)s  %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = (
            False  # Prevent the modal client from double-logging.
        )
    return logger
