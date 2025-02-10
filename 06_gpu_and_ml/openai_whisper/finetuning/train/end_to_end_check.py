"""
A full fine-tuning run on GPUs takes multiple hours, but we
want to be able to validate changes quickly while coding.

This module contains an end-to-end test that runs only 1 step of training,
before testing that the partially trained model can be serialized, saved to
persistent storage, and then downloaded locally for inference.
"""

import pathlib

from .config import app_config
from .logs import get_logger
from .train import app, persistent_volume, train
from .transcribe import whisper_transcribe_audio

logger = get_logger(__name__)


# Test model serialization and persistence by starting a new remote
# function that reads back the model files from the temporary network file system disk
# and does a single sentence of translation.
#
# When doing full training runs, the saved model will be loaded in the same way
# but from a *persisted* network file system, which keeps data around even after the Modal
# ephemeral app that ran the training has stopped.


@app.function(volumes={app_config.model_dir: persistent_volume})
def test_download_and_tryout_model(run_id: str):
    from datasets import Audio, load_dataset
    from evaluate import load

    lang, lang_short = (
        "french",
        "fr",
    )  # the language doesn't matter for this test.
    model_dir = pathlib.Path(app_config.model_dir, run_id)

    # load streaming dataset and read first audio sample
    ds = load_dataset(
        app_config.dataset,
        lang_short,
        split="test",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    test_row = next(iter(ds))
    input_speech = test_row["audio"]

    predicted_transcription = whisper_transcribe_audio(
        model_dir=model_dir,
        language=lang,
        data=input_speech["array"],
        sample_rate_hz=input_speech["sampling_rate"],
    )
    expected_transcription = test_row["sentence"]
    wer = load("wer")
    wer_score = wer.compute(
        predictions=[predicted_transcription],
        references=[expected_transcription],
    )
    logger.info(
        f"{expected_transcription=}\n{predicted_transcription=}\n"
        f"Word Error Rate (WER): {wer_score}"
    )
    assert wer_score < 1.0, (
        f"Even without finetuning, a WER score of {wer_score} is far too high."
    )


# This simple entrypoint function just starts an ephemeral app run and calls
# the two test functions in sequence.
#
# Any runtime errors or assertion errors will fail the app and exit non-zero.


@app.local_entrypoint()
def run_test():
    # Test the `main.train` function by passing in test-specific configuration
    # that does only a minimal amount of training steps and saves the model
    # to the temporary (ie. ephemeral) network file system disk.
    #
    # This should take only ~1 min to run.
    train.remote(num_train_epochs=1.0, warmup_steps=0, max_steps=1)
    test_download_and_tryout_model.remote(run_id=app.app_id)
