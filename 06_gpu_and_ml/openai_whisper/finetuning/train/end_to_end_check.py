"""
A full fine-tuning run on GPUs takes multiple hours, but we
want to be able to validate changes quickly while coding.

This module contains an end-to-end test that runs only 1 step of training,
before testing that the partially trained model can be serialized, saved to
persistent storage, and then downloaded locally for inference.
"""
import pathlib

import modal
from transformers import Seq2SeqTrainingArguments

from .config import app_config, DataTrainingArguments, ModelArguments
from .__main__ import stub, train
from .logs import get_logger
from .transcribe import whisper_transcribe_audio

test_volume = modal.NetworkFileSystem.new()

logger = get_logger(__name__)

# Test the `main.train` function by passing in test-specific configuration
# that does only a minimal amount of training steps and saves the model
# to the temporary (ie. ephemeral) network file system disk.
#
# This remote function should take only ~1 min to run.


@stub.function(network_file_systems={app_config.model_dir: test_volume})
def test_finetune_one_step_and_save_to_vol(run_id: str):
    output_dir = pathlib.Path(app_config.model_dir, run_id)
    test_model_args = ModelArguments(
        model_name_or_path="openai/whisper-small",
        freeze_feature_encoder=False,
    )
    test_data_args = DataTrainingArguments(
        preprocessing_num_workers=16,
        max_train_samples=5,
        max_eval_samples=5,
    )

    train(
        model_args=test_model_args,
        data_args=test_data_args,
        training_args=Seq2SeqTrainingArguments(
            do_train=True,
            output_dir=output_dir,
            num_train_epochs=1.0,
            learning_rate=3e-4,
            warmup_steps=0,
            max_steps=1,
        ),
    )


# Test model serialization and persistence by starting a new remote
# function that reads back the model files from the temporary network file system disk
# and does a single sentence of translation.
#
# When doing full training runs, the saved model will be loaded in the same way
# but from a *persisted* network file system, which keeps data around even after the Modal
# ephemeral app that ran the training has stopped.


@stub.function(network_file_systems={app_config.model_dir: test_volume})
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
    assert (
        wer_score < 1.0
    ), f"Even without finetuning, a WER score of {wer_score} is far too high."


# This simple entrypoint function just starts an ephemeral app run and calls
# the two test functions in sequence.
#
# Any runtime errors or assertion errors will fail the app and exit non-zero.


def run_test() -> int:
    with stub.run() as app:
        test_finetune_one_step_and_save_to_vol.call(run_id=app.app_id)
        test_download_and_tryout_model.call(run_id=app.app_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_test())
