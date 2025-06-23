import dataclasses
import logging
import pathlib


@dataclasses.dataclass
class ModelSpec:
    name: str
    params: str


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s: %(asctime)s: %(name)s  %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


CACHE_DIR = "/cache"
# Where downloaded podcasts are stored, by guid hash.
# Mostly .mp3 files 50-100MiB.
RAW_AUDIO_DIR = pathlib.Path(CACHE_DIR, "raw_audio")
# Stores metadata of individual podcast episodes as JSON.
PODCAST_METADATA_DIR = pathlib.Path(CACHE_DIR, "podcast_metadata")
# Completed episode transcriptions. Stored as flat files with
# files structured as '{guid_hash}-{model_slug}.json'.
TRANSCRIPTIONS_DIR = pathlib.Path(CACHE_DIR, "transcriptions")
# Searching indexing files, refreshed by scheduled functions.
SEARCH_DIR = pathlib.Path(CACHE_DIR, "search")
# Location of modal checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")
# Location of web frontend assets.
ASSETS_PATH = pathlib.Path(__file__).parent / "frontend" / "dist"

transcripts_per_podcast_limit = 2

supported_parakeet_models = {
    "parakeet-tdt-0.6b-v2": ModelSpec(
        name="nvidia/parakeet-tdt-0.6b-v2", params="600M"
    ),
}

DEFAULT_MODEL = supported_parakeet_models["parakeet-tdt-0.6b-v2"]
