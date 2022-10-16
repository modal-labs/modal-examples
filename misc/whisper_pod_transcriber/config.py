import dataclasses
import pathlib


@dataclasses.dataclass
class ModelSpec:
    name: str
    params: str
    relative_speed: int  # Higher is faster


CACHE_DIR = "/cache"
# Where downloaded podcasts are stored, by guid hash.
# Mostly .mp3 files 50-100MiB.
RAW_AUDIO_DIR = pathlib.Path(CACHE_DIR, "raw_audio")
# Stores metadata of individual podcast episodes as JSON.
# nb: Should probably be called EPISODES_METADATA_DIR.
METADATA_DIR = pathlib.Path(CACHE_DIR, "metadata")
PODCAST_METADATA_DIR = pathlib.Path(CACHE_DIR, "podcast_metadata")
# Crude boolean marker of a previously processed podcast.
COMPLETED_DIR = pathlib.Path(CACHE_DIR, "completed")
# Completed episode transcriptions. Stored as flat files with
# files structured as '{guid_hash}-{model_slug}.json'.
TRANSCRIPTIONS_DIR = pathlib.Path(CACHE_DIR, "transcriptions")
# Searching indexing files, refreshed by scheduled functions.
SEARCH_DIR = pathlib.Path(CACHE_DIR, "search")
# Location of modal checkpoint.
MODEL_DIR = pathlib.Path(CACHE_DIR, "model")
# Location of web frontend assets.
ASSETS_PATH = pathlib.Path(__file__).parent / "whisper_frontend" / "dist"

transcripts_per_podcast_limit = 2

supported_whisper_models = {
    "tiny.en": ModelSpec(name="tiny.en", params="39M", relative_speed=32),
    # Takes around 3-10 minutes to transcribe a podcast, depending on length.
    "base.en": ModelSpec(name="base.en", params="74M", relative_speed=16),
    "small.en": ModelSpec(name="small.en", params="244M", relative_speed=6),
    "medium.en": ModelSpec(name="medium.en", params="769M", relative_speed=2),
    # Very slow. Will take around 45 mins to 1.5 hours to transcribe.
    "large": ModelSpec(name="large", params="1550M", relative_speed=1),
}

DEFAULT_MODEL = supported_whisper_models["base.en"]
