# ---
# args: ["--test"]
# ---

# # Fine-tune Whisper to Improve Transcription on Domain-Specific Vocab

# This example demonstrates how to fine-tune an asr model
# ([whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en))
# to improve transcription accuracy for domain-specific vocabulary.

# Speech recognition foundation models work well out-of-the-box for general speech
# transcription, but can struggle with text that is not well represented in the training
# data. Fine-tuning with custom data can improve the model's ability to transcribe
# domain-specific language.

# For example, here is a sample transcription from the baseline model with no
# fine-tuning:

# ```json
# {
#   "word_error_rate": 0.67,
#   "ground_truth": "make as much deuterium and tritium as you like",
#   "prediction": "because much material and teach them what you like"
# }
# ```

# After just 2 hours of training on a small dataset (~7k samples), the model has already
# improved:

# ```json
# {
#   "word_error_rate": 0.22,
#   "ground_truth": "make as much deuterium and tritium as you like",
#   "prediction": "because much deuterium and tritium as you like"
# }
# ```


# ## Defining the environment for our Modal Functions

# We start by importing our standard library dependencies and `modal`.

# We also need an [`App`](https://modal.com/docs/guide/apps) object, which we'll use to
# define how our training application will run on Modal's cloud infrastructure.

import fastapi
import functools
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Union

import modal

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App(name="example-whisper-fine-tune")

# ### Set up the container image

# We define the environment where our functions will run by building up a base
# [container `Image`](https://modal.com/docs/guide/images)
# with our dependencies using `Image.pip_install`. We also set environment variables
# here using `Image.env`, and include a local Python module we'll want available at
# runtime using `Image.add_local_python_source`.

CACHE_DIR = "/cache"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate==1.8.1",
        "datasets==3.6.0",
        "evaluate==0.4.5",
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]==0.33.4",
        "jiwer==4.0.0",
        "librosa==0.11.0",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==4.53.2",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads from Hugging Face
            "HF_HOME": CACHE_DIR,
        }
    )
    .add_local_python_source("english_spelling_mapping")  # For text normalization
)

# Next we'll import the dependencies we need for the code that will run on Modal.

# The `image.imports()` context manager ensures these imports are available
# when our Modal functions run, but don't need to be installed locally.

with image.imports():
    import datasets
    import evaluate
    import librosa
    import torch
    import transformers
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    from english_spelling_mapping import english_spelling_mapping

# ### Storing data on Modal

# We use
# [Modal Volumes](https://modal.com/docs/guide/volumes)
# for anything we want to persist across function calls. In this case, we'll create
# a cache volume for storing Hugging Face downloads for faster subsequent loads,
# and an output Volume for saving our model and metrics after training.

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
output_volume = modal.Volume.from_name(
    "fine-tune-asr-example",  # TODO: rename to match examples repo
    create_if_missing=True,
)
OUTPUT_DIR = "/outputs"
volumes = {CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume}

# ## Calling a Modal function from the command line

# The easiest way to invoke our training Function is by creating a `local_entrypoint` --
# our `main` function that runs locally and provides a command-line interface to trigger
# training on Modal's cloud infrastructure.

# This will allow us to run this example with:

# ```bash
# modal run fine_tune_asr.py
# ```

# Arguments passed to this function are turned in to CLI arguments automagically. For
# example, adding `--test` will run a single step of training for end-to-end testing.

# ```bash
# modal run fine_tune_asr.py --test
# ```


@app.local_entrypoint()
def main(test: bool = False):
    """Run Whisper fine-tuning on Modal."""
    if test:  # for quick e2e test
        config = Config(
            dataset_subset="xs",
            num_train_epochs=1.0,
            warmup_steps=0,
            max_steps=1,
        )
    else:
        config = Config()

    train.remote(config)


# ## Defining our training Function

# Training ML models often requires a lot of configuration. We'll use a `dataclass` to
# collect some of these parameters in one place.

# For this example, we'll use the "Science and Technology" subset of the
# [GigaSpeech (small)](https://huggingface.co/datasets/speechcolab/gigaspeech)
# dataset. This is enough data to see the model improve on scientific terms in just a
# few epochs.

# GigaSpeech is a [gated model](https://huggingface.co/docs/hub/en/models-gated), so
# you'll need to accept the terms on the
# [dataset card](https://huggingface.co/datasets/speechcolab/gigaspeech)
# and create a [Hugging Face Secret](https://modal.com/secrets/) to download it.


@dataclass
class Config:
    """Training configuration."""

    run_id: str = "whisper-fine-tune"  # Name used for saving and loading

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
    batch_size: int = 64
    learning_rate: float = 1e-5


# We run evals before and after training to establish a baseline and see how much we
# improved.

# The `@app.function` decorator is where we attach infrastructure and define how our
# Function runs on Modal. Here we tell the Function to use our `Image`, specify the GPU,
# attach the Volumes we created earlier, add our access token, and set a timeout.


@app.function(
    image=image,
    gpu="H100!",
    volumes=volumes,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=3 * HOURS,
)
def train(
    config: Config,
):
    training_args = transformers.Seq2SeqTrainingArguments(
        length_column_name="input_length",
        output_dir=Path(OUTPUT_DIR) / config.run_id,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
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
    print(f"Starting training. Weights will be saved to '{training_args.output_dir}'")

    print(f"Loading model: {config.model_name}")
    feature_extractor = transformers.WhisperFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
    )
    tokenizer = transformers.WhisperTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
    )
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
    )

    print(f"Loading dataset: {config.dataset_name} {config.dataset_subset}")
    dataset = datasets.load_dataset(
        config.dataset_name,
        config.dataset_subset,
        split="train",  # The test and val splits don't have category labels
        num_proc=os.cpu_count(),
        trust_remote_code=True,
    )

    print("Preparing data")
    # Filter to only include samples from our target category (Science and Technology)
    dataset = dataset.select(
        [i for i, c in enumerate(dataset["category"]) if c == config.dataset_category]
    )

    # Keep only the columns we need: audio data and text transcription
    dataset = dataset.select_columns(["text", "audio"])

    # Split the filtered dataset into train/validation sets
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = config.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = config.min_duration_in_seconds * feature_extractor.sampling_rate
    model_input_name = feature_extractor.model_input_names[0]

    # Apply preprocessing in parallel
    dataset = dataset.map(
        functools.partial(
            prepare_dataset,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            model_input_name=model_input_name,
        ),
        remove_columns=next(iter(dataset.values())).column_names,
        num_proc=os.cpu_count(),
        desc="Preprocessing dataset",
    )

    # Filter out audio clips that are too short or too long
    dataset = dataset.filter(
        lambda length: length > min_input_length and length < max_input_length,
        num_proc=os.cpu_count(),
        input_columns=["input_length"],
    )

    normalizer = EnglishTextNormalizer(english_spelling_mapping)
    metric = evaluate.load("wer")

    # Create a processor that combines the feature extractor and tokenizer
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
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=feature_extractor,
        data_collator=data_collator,
        compute_metrics=functools.partial(
            compute_metrics,
            tokenizer=tokenizer,
            normalizer=normalizer,
            metric=metric,
        ),
    )

    print("Running evals before training to establish a baseline")
    metrics = trainer.evaluate(
        metric_key_prefix="baseline",
        max_length=training_args.generation_max_length,
        num_beams=training_args.generation_num_beams,
    )
    trainer.log_metrics("baseline", metrics)
    trainer.save_metrics("baseline", metrics)

    print("Starting training loop!")
    train_result = trainer.train()

    # Save the model weights, tokenizer, and feature extractor
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    feature_extractor.save_pretrained(training_args.output_dir)

    # Log training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset["train"])
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
    metrics["eval_samples"] = len(dataset["test"])

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    output_volume.commit()  # Ensure everything is saved to the Volume

    print(f"\nTraining complete! Model saved to '{training_args.output_dir}'")


# ## Serving our new model

# Once fine-tuning is complete, Modal makes it incredibly easy to create an API for our
# new model. We can define both our inference function and our endpoint using a Modal
# [Cls](https://modal.com/docs/reference/modal.Cls).
# This will allow us to take advantage of
# [lifecycle hooks](https://modal.com/docs/guide/lifecycle-functions)
# to load the model just once on container startup using the `@modal.enter` decorator.
# We can use
# [modal.fastapi_endpoint](https://modal.com/docs/reference/modal.fastapi_endpoint)
# to expose our inference function as a web endpoint.

# You can deploy this endpoint with:

# ```bash
# modal deploy fine_tune_asr.py
# ```

# Note: you can specify which model to load by passing the `run_id` as a query
# parameter when calling the endpoint. We set `run_id` in our `Config` above, and it's
# the name of the output directory where the model was saved.

# Here's an example of how to use this endpoint to transcribe an audio file:

# ```bash
# curl -X 'POST' \
# 'https://your-workspace-name--example-whisper-fine-tune-inference-web.modal.run/?run_id=whisper-fine-tune' \
# -H 'accept: application/json' \
# -H 'Content-Type: multipart/form-data' \
# -F 'audio_file=@your-audio-file.wav;type=audio/wav'
# ```


@app.cls(
    image=image,
    gpu="H100",
    timeout=10 * MINUTES,
    # scaledown_window=10 * MINUTES,
    volumes=volumes,
)
class Inference:
    run_id: str = modal.parameter(default=Config().run_id)

    @modal.enter()
    def load_model(self):
        """Load the model and processor on container startup."""

        model = f"{OUTPUT_DIR}/{self.run_id}" if "/" not in self.run_id else self.run_id
        print(f"Loading model from {model}")
        self.processor = transformers.WhisperProcessor.from_pretrained(model)
        self.model = transformers.WhisperForConditionalGeneration.from_pretrained(model)
        self.model.config.forced_decoder_ids = None

    @modal.method()
    def transcribe(
        self,
        audio_bytes: bytes,
    ) -> str:
        # Resample audio to match the model's sample rate
        model_sample_rate = self.processor.feature_extractor.sampling_rate
        audio_data, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)

        audio_dataset = datasets.Dataset.from_dict(
            {"audio": [{"array": audio_data, "sampling_rate": sample_rate}]}
        ).cast_column("audio", datasets.Audio(sampling_rate=model_sample_rate))

        # Audio -> features (log-mel spectrogram)
        row = next(iter(audio_dataset))
        input_features = self.processor(
            row["audio"]["array"],
            sampling_rate=row["audio"]["sampling_rate"],
            return_tensors="pt",
        ).input_features

        # generate tokens -> decode to text
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return transcription

    @modal.fastapi_endpoint(method="POST", docs=True)
    def web(
        self,
        audio_file: Annotated[bytes, fastapi.File()],
    ) -> dict[str, str]:
        """Defines an endpoint for calling inference."""

        transcription = self.transcribe.local(  # run in the same container
            audio_bytes=audio_file,
        )
        return {"transcription": transcription}


# ## Addenda

# The remainder of this code is support code, unrelated to running this example on
# Modal.


def prepare_dataset(batch, feature_extractor, tokenizer, model_input_name):
    """Convert audio to features and text to tokens."""
    sample = batch["audio"]
    inputs = feature_extractor(
        sample["array"],
        sampling_rate=sample["sampling_rate"],
    )
    batch[model_input_name] = inputs.get(model_input_name)[0]
    batch["input_length"] = len(sample["array"])

    # Normalize text: replace punctuation tags with normal punctuation, lowercase
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


def compute_metrics(pred, tokenizer, normalizer, metric):
    """Compute Word Error Rate between predictions and ground truth."""
    pred_ids = pred.predictions

    # Replace padding tokens with proper pad token ID
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels back to text
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    norm_pred_str = [normalizer(s).strip() for s in pred_str]

    label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
    norm_label_str = [normalizer(s).strip() for s in label_str]

    # Calculate Word Error Rate
    wer = metric.compute(predictions=norm_pred_str, references=norm_label_str)
    return {"wer": wer}


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
        self, features: list[dict[str, Union[list[int], "torch.Tensor"]]]
    ) -> dict[str, "torch.Tensor"]:
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

        # Remove start token if tokenizer added it - model will add it during training
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
