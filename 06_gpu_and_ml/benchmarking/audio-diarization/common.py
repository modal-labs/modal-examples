import modal

# Model names + configs
PARAKEET_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"
WHISPERX_MODEL_NAME = "large-v2"

# App + volume
APP_NAME = "audio-diarization-benchmarking-app"
MODEL_CACHE_VOLUME_NAME = "audio-diarization-model-cache"
DATASET_VOLUME_NAME = "audio-diarization-benchmarking-data"

app = modal.App(APP_NAME)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
model_cache = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)


# Constants
COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "ERROR": "\033[91m",
    "ENDC": "\033[0m",
}
