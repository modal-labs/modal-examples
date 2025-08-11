# ---
# args: ["--test"]
# ---

# # Fine-tune Whisper to Improve Transcription on Domain-Specific Vocab

# This example demonstrates how to fine-tune an ASR model
# ([whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en))
# and deploy it for inference using Modal.

# Speech recognition models work well out-of-the-box for general speech transcription,
# but can struggle with examples that are not well represented in the training data -
# like proper nouns, technical jargon, and industry-specific terms. Fine-tuning with
# examples of domain-specific vocabulary can improve transcription of these terms.

# For example, here is a sample transcription from the baseline model with no
# fine-tuning:

# |                  | Transcription                                                 |
# |------------------|---------------------------------------------------------------|
# | **Ground Truth** | "deuterium you put into one element you make a new element"   |
# | **Prediction**   | "the theorem you put into one element you make a new element" |

# After just 1.5 hours of training on a small dataset (~7k samples), the model has
# already improved:

# |                  | Transcription                                               |
# |------------------|-------------------------------------------------------------|
# | **Ground Truth** | "deuterium you put into one element you make a new element" |
# | **Prediction**   | "deuterium you put into one element you make a new element" |

# We'll use the "small" subset of "Science and Technology" from the
# [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech)
# dataset, which is enough data to see the model improve on scientific terms in just a
# few epochs.

# Note: GigaSpeech is a
# [gated model](https://huggingface.co/docs/hub/en/models-gated),
# so you'll need to accept the terms on the
# [dataset card](https://huggingface.co/datasets/speechcolab/gigaspeech)
# and create a
# [Hugging Face Secret](https://modal.com/secrets/)
# to download it.

# ## Setup

# We start by importing our standard library dependencies, `fastapi`, and `modal`.

# We also need an [`App`](https://modal.com/docs/guide/apps) object, which we'll use to
# define how our training application will run on Modal's cloud infrastructure.

import functools
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Union

import fastapi
import modal

MINUTES = 60
HOURS = 60 * MINUTES

app = modal.App(name="example-whisper-fine-tune")

# ### Set up the container image

# We define the environment where our functions will run by building up a base
# [container `Image`](https://modal.com/docs/guide/images)
# with our dependencies using `Image.uv_pip_install`. We also set environment variables
# here using `Image.env`, like the Hugging Face cache directory.

CACHE_DIR = "/cache"
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate==1.8.1",
        "datasets==3.6.0",
        "evaluate==0.4.5",
        "fastapi[standard]==0.116.1",
        "huggingface_hub[hf_transfer]==0.33.4",
        "jiwer==4.0.0",
        "librosa==0.11.0",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "transformers==4.53.2",
        "whisper_normalizer==0.1.12",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads from Hugging Face
            "HF_HOME": CACHE_DIR,
        }
    )
)

# Next we'll import the dependencies we need for the code that will run on Modal.

# The `image.imports()` context manager ensures these imports are available when our
# Functions run in the cloud, without the need to install the dependencies locally.

with image.imports():
    import datasets
    import evaluate
    import librosa
    import torch
    import transformers
    from whisper_normalizer.english import EnglishTextNormalizer

# ### Storing data on Modal

# We use
# [Modal Volumes](https://modal.com/docs/guide/volumes)
# for data we want to persist across function calls. In this case, we'll create a cache
# Volume for storing Hugging Face downloads for faster subsequent loads, and an output
# Volume for saving our model and metrics after training.

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
output_volume = modal.Volume.from_name(
    "fine-tune-asr-example-volume",
    create_if_missing=True,
)
OUTPUT_DIR = "/outputs"
volumes = {CACHE_DIR: cache_volume, OUTPUT_DIR: output_volume}

# ## Training

# We use a `dataclass` to collect some of the training parameters in one place. Here we
# set `model_output_name` which is the directory on the Volume where our model will be
# saved, and where we'll load it from when deploying the model for inference.


@dataclass
class Config:
    """Training configuration."""

    model_output_name: str = "whisper-fine-tune"  # Name used for saving and loading

    # Model config
    model_name: str = "openai/whisper-tiny.en"

    # Dataset config
    dataset_name: str = "speechcolab/gigaspeech"
    dataset_subset: str = "s"  # "xs" for testing, "m", "l", "xl" for more data
    dataset_split: str = "train"  # The test and val splits don't have category labels
    dataset_category: int = 15  # "Science and Technology"
    max_duration_in_seconds: float = 20.0
    min_duration_in_seconds: float = 0.0

    # Training config
    num_train_epochs: int = 5
    warmup_steps: int = 400
    max_steps: int = -1
    batch_size: int = 64
    learning_rate: float = 1e-5
    eval_strategy: str = "epoch"


# ### Defining our training Function

# The training Function does the following:
# 1. Load the pre-trained model, along with the feature extractor and tokenizer
# 2. Load the dataset -> select our training category -> extract features for training
# 3. Run baseline evals
# 4. 🚂 Train!
# 5. Save the fine-tuned model to the Volume
# 6. Run final evals

# We run evals before and after training to establish a baseline and see how much the
# model improved. The most common way to measure the performance of speech recognition
# models is "word error rate" (WER):

# `WER = (substitutions + deletions + insertions) / total words`.

# The `@app.function` decorator is where we attach infrastructure and define how our
# Function runs on Modal. Here we tell the Function to use our `Image`, specify the GPU,
# attach the Volumes we created earlier, add our access token, and set a timeout.


@app.function(
    image=image,
    gpu="H100",
    volumes=volumes,
    secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
    timeout=3 * HOURS,
)
def train(
    config: Config,
):
    """Loads data and trains the model."""

    # Setting args for the Hugging Face trainer
    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=Path(OUTPUT_DIR) / config.model_output_name,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        eval_strategy=config.eval_strategy,
        fp16=True,
        group_by_length=True,
        length_column_name="input_length",
        predict_with_generate=True,
        generation_max_length=40,
        generation_num_beams=1,
    )

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
    dataset = (
        datasets.load_dataset(
            config.dataset_name,
            config.dataset_subset,
            split=config.dataset_split,
            num_proc=os.cpu_count(),
            trust_remote_code=True,
        )
        if config.dataset_name is not None
        else get_test_dataset(config)
    )

    print("Preparing data")
    max_input_length = config.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = config.min_duration_in_seconds * feature_extractor.sampling_rate

    # Remove samples that are not from our target category (Science and Technology)
    # Remove audio clips that are too short or too long
    dataset = dataset.filter(
        functools.partial(
            filter_dataset,
            dataset_category=config.dataset_category,
            max_input_length=max_input_length,
            min_input_length=min_input_length,
        ),
        input_columns=["category", "audio"],
        num_proc=os.cpu_count(),
    )

    # Extract audio features and tokenize labels
    dataset = dataset.map(
        functools.partial(
            prepare_dataset,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            model_input_name=feature_extractor.model_input_names[0],
        ),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
        desc="Feature extract + tokenize",
    )

    # Split the filtered dataset into train/validation sets
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

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
    normalizer = EnglishTextNormalizer()
    metric = evaluate.load("wer")

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

    print(f"Starting training! Weights will be saved to '{training_args.output_dir}'")
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


# ## Calling a Modal function from the command line

# The easiest way to invoke our training Function is by creating a `local_entrypoint` --
# our `main` function that runs locally and provides a command-line interface to trigger
# training on Modal's cloud infrastructure.


@app.local_entrypoint()
def main(test: bool = False):
    """Run Whisper fine-tuning on Modal."""
    if test:  # for quick e2e test
        config = Config(
            dataset_name=None,
            num_train_epochs=1.0,
            warmup_steps=0,
            max_steps=1,
        )
    else:
        config = Config()

    train.remote(config)


# This will allow us to run this example with:

# ```bash
# modal run fine_tune_asr.py
# ```

# Arguments passed to this function are turned in to CLI arguments automagically. For
# example, adding `--test` will run a single step of training for end-to-end testing.

# ```bash
# modal run fine_tune_asr.py --test
# ```

# Training will take ~1.5 hours, and will log WER and other metrics throughout the
# run.

# Here are a few more examples of terms the model predicted correctly after fine-tuning:

# | **Base Model** | **Fine-tuned**  |
# |----------------|-----------------|
# | and pm package | npm package     |
# | teach them     | tritium         |
# | chromebox      | chromevox       |
# | purposes       | porpoises       |
# | difsoup        | div soup        |
# | would you      | widget          |

# ## Deploying our fine-tuned model for inference

# Once fine-tuning is complete, Modal makes it incredibly easy to deploy our new model.
# We can define both our inference function and an endpoint using a Modal
# [Cls](https://modal.com/docs/reference/modal.Cls).
# This will allow us to take advantage of
# [lifecycle hooks](https://modal.com/docs/guide/lifecycle-functions)
# to load the model just once on container startup using the `@modal.enter` decorator.
# We can use
# [modal.fastapi_endpoint](https://modal.com/docs/reference/modal.fastapi_endpoint)
# to expose our inference function as a web endpoint.


@app.cls(
    image=image,
    gpu="H100",
    timeout=10 * MINUTES,
    # scaledown_window=10 * MINUTES,
    volumes=volumes,
)
class Inference:
    model_name: str = modal.parameter(default=Config().model_output_name)

    @modal.enter()
    def load_model(self):
        """Load the model and processor on container startup."""

        model = f"{OUTPUT_DIR}/{self.model_name}"
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


# Deploy it with:

# ```bash
# modal deploy fine_tune_asr.py
# ```

# Note: you can specify which model to load by passing the `model_name` as a
# query parameter when calling the endpoint. This defaults to `model_output_name`, which
# we set in our `Config` above, and is the name of the directory where our model
# was saved.

# Here's an example of how to use this endpoint to transcribe an audio file:

# ```bash
# curl -X 'POST' \
# 'https://your-workspace-name--example-whisper-fine-tune-inference-web.modal.run/?model_name=whisper-fine-tune' \
# -H 'accept: application/json' \
# -H 'Content-Type: multipart/form-data' \
# -F 'audio_file=@your-audio-file.wav;type=audio/wav'
# ```

# ## Support code


def get_test_dataset(config, length=5):
    return datasets.Dataset.from_dict(
        {
            "text": ["Modal"] * length,
            "audio": [{"array": [1.0] * 16000, "sampling_rate": 16000}] * length,
            "category": [config.dataset_category] * length,
        }
    )


def filter_dataset(
    category, audio, dataset_category, max_input_length, min_input_length
):
    return (
        category == dataset_category
        and len(audio["array"]) > min_input_length
        and len(audio["array"]) < max_input_length
    )


def prepare_dataset(batch, feature_extractor, tokenizer, model_input_name):
    """Batched: convert audio to features and text to token IDs."""
    audio_arrays = [s["array"] for s in batch["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
    )
    batch[model_input_name] = inputs.get(model_input_name)
    batch["input_length"] = [len(s["array"]) for s in batch["audio"]]

    normalized = [
        t.replace(" <COMMA>", ",")
        .replace(" <PERIOD>", ".")
        .replace(" <QUESTIONMARK>", "?")
        .replace(" <EXCLAMATIONPOINT>", "!")
        .lower()
        .strip()
        for t in batch["text"]
    ]
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
