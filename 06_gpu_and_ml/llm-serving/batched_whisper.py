# # Fast Whisper inference using dynamic batching
#
# In this example, we demonstrate how to run batched inference for [OpenAI's Whisper](https://openai.com/index/whisper/),
# a speech recognition model. By batching multiple audio samples together or batching chunks of a single audio sample,
# we can achieve up to a 2.5x speedup in inference throughput with on an A100.
#
# We will be running the [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3) model.
# To run [any of the other HuggingFace Whisper models](https://huggingface.co/models?search=openai/whisper),
# simply replace the `MODEL_NAME` and `MODEL_REVISION` variables.
#
# ## Setup
#
# First, we import the Modal client and define the model that we want to serve.

import asyncio
import os
import time

import modal
from datasets import load_dataset

MODEL_DIR = "/model"
MODEL_NAME = "openai/whisper-large-v3"
MODEL_REVISION = "afda370583db9c5359511ed5d989400a6199dfe1"

# ## Define a container image
#
# We want to create a Modal image that has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Hugging Face. Instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# We can download the model to a specific directory using the Hugging Face utility function `snapshot_download`.
#
# If you adapt this example to run another model, note that for this step to work on a
# [gated model](https://huggingface.co/docs/hub/en/models-gated),
# the `HF_TOKEN` environment variable must be set and provided as a [Modal Secret](https://modal.com/secrets).

def download_model_to_image(model_dir, model_name, model_revision):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        revision=model_revision,
    )
    move_cache()

# ### Image Definition
#
# We’ll start with Modal's baseline `debian_slim` image and install the relevant libraries.
# Then we’ll use `run_function` with `download_model_to_image` to write the model into the container image.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.39.3",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "datasets==2.20.0",
        "accelerate==0.33.0",
    )
    # Use the barebones `hf-transfer` package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        timeout=60 * 20,
        kwargs={
            "model_dir": MODEL_DIR,
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
        },
    )
)

app = modal.App("example-whisper-batched-inference", image=image)

# ## The model class
#
# The inference function is best represented using Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions),
# with a `load_model` method decorated with `@modal.enter`. This enables us to load the model into memory just once,
# every time a container starts up, and keep it cached on the GPU for subsequent invocations of the function.
#
# We also define a `transcribe` method that uses the `@modal.batched` decorator to enable dynamic batching.
# This allows us to invoke the function with individual audio samples, and the function will automatically batch them
# together before running inference. The `max_batch_size` parameter limits the batch size to a maximum of 128 audio samples
# at a time. The `wait_ms` parameter sets the maximum time to wait for more inputs before running the batched transcription.
#
# We selected a batch size of 128 because it is the largest power of 2 that fits within the 40GB A100 GPU memory.
# This number will vary depending on the model and the GPU you are using. To tune the `wait_ms` parameter, you can set it to
# `(targeted latency) - (execution time)`. Most applications have a targeted latency, and this allows the latency of
# any request to stay within that limit.
#
# Hint: Try using an H100 if you've got a large model or big batches!

GPU_CONFIG = modal.gpu.A100(count=1)  # 40GB A100 by default

@app.cls(gpu=GPU_CONFIG, concurrency_limit=1)
class Model:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to("cuda")

        # Create a pipeline for preprocessing speech data and transcribing it
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda"
        )

    @modal.batched(max_batch_size=128, wait_ms=4000)
    def transcribe(self, audio_samples):
        transcription = self.pipeline(audio_samples, batch_size=len(audio_samples))
        return transcription


# ## Run the model
#
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps)
# to call our remote function sequentially for a list of inputs. You can run this locally with
# `modal run batched_whisper.py`.
#
# In this example, we use the [librispeech_asr_dummy dataset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
# from Hugging Face's Datasets library to test the model.
#
# We use [`map.aio`](/docs/reference/modal.Function#map) to asynchronously map over the audio files.
# This allows us to invoke the batched transcription method on each audio sample in parallel.

@app.local_entrypoint()
async def main():
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    batched_whisper = Model()
    async for transcription in batched_whisper.transcribe.map.aio(ds["audio"][:20]):
        print("Transcription for audio 📻", transcription)
