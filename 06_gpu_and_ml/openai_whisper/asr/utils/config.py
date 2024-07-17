import logging

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from typing import Optional, Literal

logger = logging.getLogger(__name__)

audio_types = {
    "audio/x-flac",
    "audio/flac",
    "audio/mpeg",
    "audio/x-mpeg-3",
    "audio/wave",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/x-audio",
    "audio/webm",
    "audio/webm;codecs=opus",
    "audio/AMR",
    "audio/amr",
    "audio/AMR-WB",
    "audio/AMR-WB+",
    "audio/m4a",
    "audio/x-m4a"
}


class ModelSettings(BaseSettings):
    asr_model: str
    assistant_model: Optional[str] = None
    diarization_model: Optional[str] = None
    hf_token: Optional[str] = None


class InferenceConfig(BaseModel):
    task: Literal["transcribe", "translate"] = "transcribe"
    batch_size: int = 24
    chunk_length_s: int = 30
    sampling_rate: int = 16000
    assisted: bool = False
    language: Optional[str] = None
    num_speakers: Optional[int] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None


model_settings = ModelSettings(
    asr_model="openai/whisper-small",
    assistant_model="distil-whisper/distil-small.en",
    diarization_model="pyannote/speaker-diarization-3.1",
    hf_token="..."
)

logger.info(f"asr model: {model_settings.asr_model}")
logger.info(f"assist model: {model_settings.assistant_model}")
logger.info(f"diar model: {model_settings.diarization_model}")
