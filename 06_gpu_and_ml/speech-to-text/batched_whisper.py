# # Fast Whisper inference using dynamic batching

# In this example, we demonstrate how to run [dynamically batched inference](https://modal.com/docs/guide/dynamic-batching)
# for OpenAI's speech recognition model, [Whisper](https://openai.com/index/whisper/), on Modal.
# Batching multiple audio samples together or batching chunks of a single audio sample can help to achieve a 2.8x increase
# in inference throughput on an A10G!

# We will be running the [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3) model.
# To run [any of the other HuggingFace Whisper models](https://huggingface.co/models?search=openai/whisper),
# simply replace the `MODEL_NAME` and `MODEL_REVISION` variables.

# ## Setup

# Let's start by importing the Modal client and defining the model that we want to serve.


from typing import Optional

import modal

MODEL_DIR = "/model"
MODEL_NAME = "openai/whisper-large-v3"
MODEL_REVISION = "afda370583db9c5359511ed5d989400a6199dfe1"


# ## Define a container image

# We’ll start with Modal's baseline `debian_slim` image and install the relevant libraries.

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.47.1",
        "hf-transfer==0.1.8",
        "huggingface_hub==0.27.0",
        "librosa==0.10.2",
        "soundfile==0.12.1",
        "accelerate==1.2.1",
        "datasets==3.2.0",
    )
    # Use the barebones `hf-transfer` package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_CACHE": MODEL_DIR})
)

model_cache = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
app = modal.App(
    "example-batched-whisper",
    image=image,
    volumes={MODEL_DIR: model_cache},
)

# ## Caching the model weights

# We'll define a function to download the model and cache it in a volume.
# You can `modal run` against this function prior to deploying the App.


@app.function()
def download_model():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    snapshot_download(
        MODEL_NAME,
        ignore_patterns=["*.pt", "*.bin"],  # Using safetensors
        revision=MODEL_REVISION,
    )
    move_cache()


# ## The model class

# The inference function is best represented using Modal's [class syntax](https://modal.com/docs/guide/lifecycle-functions).

# We define a `@modal.enter` method to load the model when the container starts, before it picks up any inputs.
# The weights will be loaded from the Hugging Face cache volume so that we don't need to download them when
# we start a new container. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

# We also define a `transcribe` method that uses the `@modal.batched` decorator to enable dynamic batching.
# This allows us to invoke the function with individual audio samples, and the function will automatically batch them
# together before running inference. Batching is critical for making good use of the GPU, since GPUs are designed
# for running parallel operations at high throughput.

# The `max_batch_size` parameter limits the maximum number of audio samples combined into a single batch.
# We used a `max_batch_size` of `64`, the largest power-of-2 batch size that can be accommodated by the 24 A10G GPU memory.
# This number will vary depending on the model and the GPU you are using.

# The `wait_ms` parameter sets the maximum time to wait for more inputs before running the batched transcription.
# To tune this parameter, you can set it to the target latency of your application minus the execution time of an inference batch.
# This allows the latency of any request to stay within your target latency.


@app.cls(
    gpu="a10g",  # Try using an A100 or H100 if you've got a large model or need big batches!
    max_containers=10,  # default max GPUs for Modal's free tier
)
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

        self.model.generation_config.language = "<|en|>"

        # Create a pipeline for preprocessing and transcribing speech data
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=torch.float16,
            device="cuda",
        )

    @modal.batched(max_batch_size=64, wait_ms=1000)
    def transcribe(self, audio_samples):
        import time

        start = time.monotonic_ns()
        print(f"Transcribing {len(audio_samples)} audio samples")
        transcriptions = self.pipeline(audio_samples, batch_size=len(audio_samples))
        end = time.monotonic_ns()
        print(
            f"Transcribed {len(audio_samples)} samples in {round((end - start) / 1e9, 2)}s"
        )
        return transcriptions


# ## Transcribe a dataset

# In this example, we use the [librispeech_asr_dummy dataset](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy)
# from Hugging Face's Datasets library to test the model.

# We use [`map.aio`](https://modal.com/docs/reference/modal.Function#map) to asynchronously map over the audio files.
# This allows us to invoke the batched transcription method on each audio sample in parallel.


@app.function()
async def transcribe_hf_dataset(dataset_name):
    from datasets import load_dataset

    print("📂 Loading dataset", dataset_name)
    ds = load_dataset(dataset_name, "clean", split="validation")
    print("📂 Dataset loaded")
    batched_whisper = Model()
    print("📣 Sending data for transcription")
    async for transcription in batched_whisper.transcribe.map.aio(ds["audio"]):
        yield transcription


# ## Run the model

# We define a [`local_entrypoint`](https://modal.com/docs/guide/apps#entrypoints-for-ephemeral-apps)
# to run the transcription. You can run this locally with `modal run batched_whisper.py`.


@app.local_entrypoint()
async def main(dataset_name: Optional[str] = None):
    if dataset_name is None:
        dataset_name = "hf-internal-testing/librispeech_asr_dummy"
    for result in transcribe_hf_dataset.remote_gen(dataset_name):
        print(result["text"])
