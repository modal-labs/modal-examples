import os
from typing import TYPE_CHECKING

from .logs import get_logger

if TYPE_CHECKING:
    from numpy import ndarray

logger = get_logger(__name__)


def whisper_transcribe_local_file(
    model_dir: os.PathLike,
    language: str,
    filepath: os.PathLike,
    sample_rate_hz: int,
) -> str:
    """Convenience function for transcribing a single local audio file with a Whisper model already saved to disk."""
    from datasets import Audio, Dataset

    audio_dataset = Dataset.from_dict({"audio": [str(filepath)]}).cast_column(
        "audio", Audio(sampling_rate=sample_rate_hz)
    )
    row = next(iter(audio_dataset))
    return whisper_transcribe_audio(
        model_dir,
        language,
        data=row["audio"]["array"],
        sample_rate_hz=row["audio"]["sampling_rate"],
    )


def whisper_transcribe_audio(
    model_dir: os.PathLike,
    language: str,
    data: "ndarray",
    sample_rate_hz: int,
) -> str:
    """Transcribes a single audio sample with a Whisper model, for demonstration purposes."""
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    # load model and processor
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )
    input_features = processor(
        data,
        sampling_rate=sample_rate_hz,
        return_tensors="pt",
    ).input_features

    # generate token ids
    predicted_ids = model.generate(
        input_features, forced_decoder_ids=forced_decoder_ids
    )
    # decode token ids to text
    predicted_transcription = processor.batch_decode(
        predicted_ids, skip_special_tokens=True
    )[0]
    return predicted_transcription
