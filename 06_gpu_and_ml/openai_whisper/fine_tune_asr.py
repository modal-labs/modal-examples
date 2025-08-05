# ---
# args: ["--test"]
# ---

# # Fine-tune Whisper to Improve Domain-Specific Transcription

# This example demonstrates how to fine-tune an OpenAI Whisper model to improve the
# model's abillity to transcribe domain-specific vocabulary.

import os
import time
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from typing import Any, Union

import modal

OUTPUT_DIR = "/outputs"
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

CACHE_DIR = "/cache"
output_volume = modal.Volume.from_name(
    "fine-tune-asr-example",  # TODO: rename to match examples repo
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate",
        "datasets==3.6.0",
        "evaluate",
        "huggingface_hub[hf_transfer]",
        "jiwer",
        "librosa",
        "torch",
        "torchaudio",
        "transformers",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": CACHE_DIR,
        }
    )
    .pip_install("ipdb", "IPython")  # TODO: REMOVE
    .add_local_python_source("english_spelling_mapping")
)

with image.imports():
    import datasets
    import evaluate
    import torch
    import transformers
    from transformers.models.whisper.english_normalizer import (
        EnglishTextNormalizer,
    )
    from transformers.trainer_utils import get_last_checkpoint, is_main_process

    from english_spelling_mapping import english_spelling_mapping


app = modal.App(
    name="example-whisper-fine-tune",
)

# TODO: REMOVE
app.set_description(os.environ.get("APP_NAME", app.name))


@dataclass
class Config:
    """Training configuration."""

    # Model config
    model_name: str = "openai/whisper-tiny.en"

    # Dataset config
    dataset_name: str = "speechcolab/gigaspeech"
    dataset_subset: str = "s"  # "xs" for testing, "m", "l", "xl" for more data
    dataset_category: int = 15  # "Science and Technology"
    max_duration_in_seconds: float = 20.0
    min_duration_in_seconds: float = 0.0

    # Training config
    num_train_epochs: int = 5
    warmup_steps: int = 400
    max_steps: int = -1


@app.local_entrypoint()
def main(test: bool = False):
    if test:
        config = Config(
            dataset_subset="xs",
            num_train_epochs=1.0,
            warmup_steps=0,
            max_steps=1,
        )
    else:
        config = Config()

    start = time.perf_counter()
    train.remote(config)
    print(f"Training took {time.perf_counter() - start:.6f} seconds")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    gpu="H100!",
    volumes={CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume},
    timeout=24 * 60 * 60,  # TODO: Tighten up
)
def train(
    config: Config,
):
    training_args = transformers.Seq2SeqTrainingArguments(
        length_column_name="input_length",
        output_dir=Path(OUTPUT_DIR) / app.app_id,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        eval_strategy="steps",
        save_total_limit=3,
        fp16=True,
        group_by_length=True,
        predict_with_generate=True,
        generation_max_length=40,
        generation_num_beams=1,
    )

    print("Starting training run")
    print(f"Starting training. Weights will be saved to '{training_args.output_dir}'")

    print("Loading models")

    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        cache_dir=CACHE_DIR,
    )
    tokenizer = transformers.WhisperTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        cache_dir=CACHE_DIR,
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        cache_dir=CACHE_DIR,
    )

    print("Loading dataset")
    ds = datasets.load_dataset(
        config.dataset_name,
        config.dataset_subset,
        split="train",  # The test and val splits
        cache_dir=CACHE_DIR,
        num_proc=os.cpu_count(),
        trust_remote_code=True,
    )

    print("Preparing data")
    # Filter to only include samples from our target category (Science and Technology)
    ds = ds.select(
        [i for i, c in enumerate(ds["category"]) if c == config.dataset_category]
    )

    # Keep only the columns we need: audio data and text transcription
    ds = ds.select_columns(["text", "audio"])

    # Split the filtered dataset into train/validation sets
    raw_datasets = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)

    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = config.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = config.min_duration_in_seconds * feature_extractor.sampling_rate
    model_input_name = feature_extractor.model_input_names[0]

    def prepare_dataset(batch):
        # process audio
        sample = batch["audio"]
        inputs = feature_extractor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
        )
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])

        # Normalize text
        normalized = (
            batch["text"]
            .replace(" <COMMA>", ",")
            .replace(" <PERIOD>", ".")
            .replace(" <QUESTIONMARK>", "?")
            .replace(" <EXCLAMATIONPOINT>", "!")
            .lower()
            .strip()
        )

        batch["labels"] = tokenizer(normalized).input_ids

        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        num_proc=os.cpu_count(),
        desc="Preprocessing dataset",
    )

    # Filter out audio clips that are too short or too long
    vectorized_datasets = vectorized_datasets.filter(
        lambda length: length > min_input_length and length < max_input_length,
        num_proc=os.cpu_count(),
        input_columns=["input_length"],
    )

    normalizer = EnglishTextNormalizer(english_spelling_mapping)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        norm_pred_str = [normalizer(s).strip() for s in pred_str]

        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        norm_label_str = [normalizer(s).strip() for s in label_str]

        wer = metric.compute(predictions=norm_pred_str, references=norm_label_str)
        return {"wer": wer}

    processor = transformers.WhisperProcessor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # Custom data collator handles batching of variable-length audio sequences
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Set up the Hugging Face trainer with all of our components
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["test"],
        processing_class=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Running evals before training to establish a baseline")
    metrics = trainer.evaluate(
        metric_key_prefix="test",
        max_length=training_args.generation_max_length,
        num_beams=training_args.generation_num_beams,
    )
    metrics["eval_samples"] = len(vectorized_datasets["test"])

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    print("Starting training loop")
    train_result = trainer.train()
    trainer.save_model()  # Saves the model, feature extractor, and tokenizer
    print(f"Model saved to '{training_args.output_dir}'")

    # Log training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(vectorized_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Final evaluation to see how much we improved
    print("Running final evals")
    metrics = trainer.evaluate(
        metric_key_prefix="test",
        max_length=training_args.generation_max_length,
        num_beams=training_args.generation_num_beams,
    )
    metrics["eval_samples"] = len(vectorized_datasets["test"])

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    output_volume.commit()  # Ensure the model and metrics are saved to the Volume

    print("Training complete!")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that pads audio features and text labels for batch training.

    Args:
        processor: WhisperProcessor combining feature extractor and tokenizer
        decoder_start_token_id: The BOS token ID for the decoder
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: list[dict[str, Union[list[int], torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        # Separate audio features and text labels since they need different padding
        model_input_name = self.processor.model_input_names[0]
        input_features = [
            {model_input_name: feature[model_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        )

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding tokens with -100 so they're ignored in loss calculation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's appended later anyways
        # TODO: Is this necessary?
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
