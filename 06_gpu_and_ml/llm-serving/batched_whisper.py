# # Fast Whisper inference using dynamic batching
#
# In this example, we demonstrate how to run batched inference for OpenAI's speech recognition model,
# [Whisper](https://openai.com/index/whisper/). Batching multiple audio samples together or batching chunks
# of a single audio sample can help to achieve a 2.5x speedup in inference throughput on an A100!
#
# We will be running the [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3) model.
# To run [any of the other HuggingFace Whisper models](https://huggingface.co/models?search=openai/whisper),
# simply replace the `MODEL_NAME` and `MODEL_REVISION` variables.
#
# ## Setup
#
# Let's start by importing the Modal client and defining the model that we want to serve.

import os

import modal

MODEL_DIR = "/model"
MODEL_NAME = "openai/whisper-large-v3"
MODEL_REVISION = "afda370583db9c5359511ed5d989400a6199dfe1"


# ## Define a container image
#
# We’ll start with Modal's baseline `debian_slim` image and install the relevant libraries.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "transformers==4.39.3",
        "hf-transfer==0.1.6",
        "huggingface_hub==0.22.2",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "accelerate==0.33.0",
        "datasets==2.20.0",
    )
    # Use the barebones `hf-transfer` package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("example-whisper-batched-inference", image=image)


# ## The model class
#
# The inference function is best represented using Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions).

# We define a `@modal.build` method to download the model and a `@modal.enter` method to load the model. This allows
# the container to download the model from HuggingFace just once when it launches, load the model into memory just once
# every time a container starts up by caching it on the GPU for subsequent invocations of the function.
#
# We also define a `transcribe` method that uses the `@modal.batched` decorator to enable dynamic batching.
# This allows us to invoke the function with individual audio samples, and the function will automatically batch them
# together before running inference.
#
# The `max_batch_size` parameter limits the maximum number of audio samples combined into a single batch.
# We used a `max_batch_size` of 128, the largest power of 2 that can be accommodated by the 40GB A100 GPU memory. This number
# will vary depending on the model and the GPU you are using.
#
# The `wait_ms` parameter sets the maximum time to wait for more inputs before running the batched transcription.
# To tune this parameter, you can set it to the target latency of your application minus the execution time of an inference batch.
# This allows the latency of any request to stay within your target latency.
#
# Hint: Try using an H100 if you've got a large model or big batches!

GPU_CONFIG = modal.gpu.A100(count=1)  # 40GB A100 by default

@app.cls(gpu=GPU_CONFIG, concurrency_limit=1)
class Model:
    @modal.build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        from transformers.utils import move_cache

        os.makedirs(MODEL_DIR, exist_ok=True)

        snapshot_download(
            MODEL_NAME,
            local_dir=MODEL_DIR,
            ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
            revision=MODEL_REVISION,
        )
        move_cache()

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

        # Create a pipeline for preprocessing and transcribing speech data
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda",
        )

    @modal.batched(max_batch_size=128, wait_ms=4000)
    def transcribe(self, audio_samples):
        transcription = self.pipeline(
            audio_samples, batch_size=len(audio_samples)
        )
        return transcription


# ## Transcribe a dataset
# In this example, we use the [librispeech_asr_dummy dataset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
# from Hugging Face's Datasets library to test the model.
#
# We use [`map.aio`](/docs/reference/modal.Function#map) to asynchronously map over the audio files.
# This allows us to invoke the batched transcription method on each audio sample in parallel.


@app.function()
async def transcribe_hf_dataset(dataset_name):
    from datasets import load_dataset

    ds = load_dataset(
        dataset_name, "clean", split="validation"
    )
    batched_whisper = Model()
    async for transcription in batched_whisper.transcribe.map.aio(
        ds["audio"]
    ):
        print("Transcription for audio 📻", transcription["text"])


# ## Run the model
#
# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps)
# to run the transcription. You can run this locally with `modal run batched_whisper.py`.


@app.local_entrypoint()
async def main():
    transcribe_hf_dataset.remote("hf-internal-testing/librispeech_asr_dummy")
