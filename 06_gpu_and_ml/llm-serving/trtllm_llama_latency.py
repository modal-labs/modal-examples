# ---
# deploy: true
# ---

# # Real-Time User Experiences with Latency-Optimized TensorRTLLM (LLaMA 3 8B)

# The [Doherty Threshold](https://lawsofux.com/doherty-threshold/) is a crucial
# concept in user experience and human-computer interaction that was identified by
# IBM researcher Walter J. Doherty in the early 1980s. His research established that
# response times under 400 milliseconds create a profound shift in how humans interact
# with technology and we've all felt this with the rise of LLMs like ChatGPT.

# In this example, we demonstrate how to use configure the TensorRT-LLM framework to serve
# Meta's LLaMA 3 8B model under this 400ms threshold using several key parameters.

# TensorRT-LLM is the Lamborghini of inference engines: it achieves seriously
# impressive latency, but only if you tune it carefully. With the default configuration
# we'll get a slow p50 latency of 1.1s, but with careful configuration, we'll bring that down
# to an astonishing 0.2s, that's more than a 5x speed up! These latencies are for running on a
# single NVIDIA H100 GPU, at [Modal's on-demand rate](https://modal.com/pricing) of ~$3.95/hr,
# that comes out to almost 50 inference calls per cent.

# ## Overview

# This guide is intended to document two things:
# the [new python API](https://nvidia.github.io/TensorRT-LLM/llm-api/) for TensorRT-LLM
# and how to use recommendations from the [performance guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/useful-build-time-flags.html)
# to optimize the engine for low latency. Be sure to check out TRTLLM's
# [examples](https://nvidia.github.io/TensorRT-LLM/llm-api-examples/) for
# use cases beyond this example, e.g. LoRA adapters.

# ### Engine building

# The first step in running TensorRT-LLM is to build an engine from a pre-trained model.
# The number of parameters for this is pretty gnarly but we'll carefully document the
# choices we made here and point you to additional resources that can help you optimize for
# your specific workload.

# Historically, this process has been done with a clunky command-line-interface (CLI),
# but things have changed for the better. The new python API is a huge improvement ergonomically and has
# all the same features as the CLI such as quantization, speculative decoding, in-flight batching,
# and much more.

# This example builds an entire service from scratch, from downloading weight tensors
# to responding to requests, and so serves as living, interactive documentation of an
# optimized TensorRT-LLM build process that deploys on Modal.

# ## Installing TensorRT-LLM

# To run TensorRT-LLM, we must first install it. Easier said than done!

# In Modal, we define [container images](https://modal.com/docs/guide/custom-container) that run our serverless workloads.
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.

# We start from an official `nvidia/cuda` image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.

import time
from pathlib import Path

import modal

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.4.1-devel-ubuntu22.04",
    add_python="3.12",  # TRT-LLM requires Python 3.12
).entrypoint([])  # remove verbose logging by base image on entry

# On top of that, we add some system dependencies of TensorRT-LLM,
# including OpenMPI for distributed communication, some core software like `git`,
# the `tensorrt_llm` package itself, and finally `flashinfer` for optimized
# [flash attention kernels](https://docs.flashinfer.ai/) that TensorRT-LLM uses.

tensorrt_image = (
    tensorrt_image.apt_install(
        "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
    )
    .pip_install(
        "tensorrt-llm==0.18.0.dev2025031100",
        "pynvml<12",  # avoid breaking change to pynvml version API
        pre=True,
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install(
        "flashinfer-python",
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.4/",
    )
)

# Note that we're doing this by [method-chaining](https://quanticdev.com/articles/method-chaining/)
# a number of calls to methods on the `modal.Image`. If you're familiar with
# Dockerfiles, you can think of this as a Pythonic interface to instructions like `RUN` and `CMD`.

# End-to-end, this step takes five minutes.
# If you're reading this from top to bottom,
# you might want to stop here and execute the example
# with `modal run trtllm_llama_latency.py`
# so that it runs in the background while you read the rest.

# ## Downloading the Model

# Next, we'll set up a few things to download the model to persistent storage and do it quickly,
# this is a latency-optimized example after all! For persistent, distributed storage, we use
# [Modal volumes](https://modal.com/docs/guide/volumes) which can be accessed from any container.
# We also set up the `HF_HOME` environment variable to point to the volume so that the model
# is cached there. Then we install `hf-transfer` to get max download throughput from Hugging Face.

volume = modal.Volume.from_name(
    "example-trtllm-inference-volume", create_if_missing=True
)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"  # fork without repo gating
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"

tensorrt_image = tensorrt_image.pip_install(
    "hf-transfer==0.1.9",
    "huggingface_hub==0.28.1",
).env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": str(MODELS_PATH),
    }
)

with tensorrt_image.imports():
    import os

    import torch
    from tensorrt_llm import LLM, SamplingParams

# ## Setting up the Engine
# ### Quantization

# The amount of GPU RAM on a single card is a tight constraint for most LLMs:
# RAM is measured in billions of bytes and models have billions of parameters.
# The performance cliff if you need to spill to CPU memory is steep,
# so all of those parameters must fit in the GPU memory,
# along with other things like the KV cache.

# The simplest way to reduce LLM inference's RAM requirements is to make the model's parameters smaller,
# to fit their values in a smaller number of bits, like four or eight. This is known as _quantization_.

# NVIDIA's Ada Lovelace/Hopper chips, like the 4090, L40S, and H100,
# are capable of native calculations in 8bit floating point numbers, so we choose that as our quantization format.
# These GPUs are capable of twice as many floating point operations per second in 8bit as in 16bit --
# about two quadrillion per second on an H100 SXM.

# So quantization buys us two things: Fast cold boots, since less data has to be moved onto the
# container and GPU, and faster inference, since we get twice the FLOPs. We'll use trtlllm's
# `QuantConfig` to specify that we want `FP8` quantization, see
# [here](https://github.com/NVIDIA/TensorRT-LLM/blob/88e1c90fd0484de061ecfbacfc78a4a8900a4ace/tensorrt_llm/models/modeling_utils.py#L184)
# for more options.

N_GPUS = 1  # Bumping this to 2 will improve latencies further but not 2x.
GPU_CONFIG = f"H100:{N_GPUS}"


def get_quant_config():
    from tensorrt_llm.llmapi import QuantConfig

    return QuantConfig(quant_algo="FP8")


# One caveat of quantization is that it's lossy, but the impact on modal quality can be
# minimized by tuning the quantization parameters given a small dataset. Typically, we
# see less than 2% degradation in evaluation metrics when using `fp8`. We'll use the
# trtllm `CalibrationConfig` class to specify the calibration dataset:


def get_calib_config():
    from tensorrt_llm.llmapi import CalibConfig

    return CalibConfig(
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=2048,
        tokenizer_max_seq_length=4096,
    )


# ### Plugins

# A little background - TensorRT-LLM is built on top of NVIDIA's TensorRT, which is a high-performance
# deep learning inference engine. TensortRT allows for custom "plugins" that allow you to
# override the default kernels TensorRT would use for certain operations. The
# General Matrix Multiply [(GEMM)](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
# plugin, for instance, utilizes NVIDIA's cuBLASLt library to provide high-performance matrix multiplication.

# We'll specify a number of plugins for our engine implementation.
# The first is [multiple profiles](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/performance/performance-tuning-guide/useful-build-time-flags.md#multiple-profiles)
# which allows trtllm to prepare multiple kernels, optimized for different input sizes,
# that will be used dynamically at generation time based on the size of the prompt.
# The second is `paged_kv_cache` which allows for breaking up the KV caches into pages to
# reduce memory fragmentation and improve efficiency.

# The last two parameters are gemm plugins optimized specifically for low latency.
# The `low_latency_gemm_swiglu_plugin` plugin fuses two Matmul operations and one
# SwiGLU operation into a single kernel which reduces GPU memory transfers
# resulting in lower latencies. Note, at the time of writing, this only works
# for `FP8` on Hopper GPUs. The `low_latency_gemm_plugin` is a
# variant of the typical gemm plugin that's latency optimized.


def get_plugin_config():
    from tensorrt_llm.plugin.plugin import PluginConfig

    return PluginConfig.from_dict(
        {
            "multiple_profiles": True,
            "paged_kv_cache": True,
            "low_latency_gemm_swiglu_plugin": "fp8",
            "low_latency_gemm_plugin": "fp8",
        }
    )


# ### Speculative Decoding

# Speculative decoding is a technique for generating multiple tokens per step,
# avoiding the LLM auto-regressive bottleneck of generating one token at a time.
# This generally works best for text that has predicable patterns, like code,
# but it's worth testing anytime latency is crtical. We'll use a simple
# speculative decoding strategy called lookahead decoding here:


def get_speculative_config():
    from tensorrt_llm.llmapi import LookaheadDecodingConfig

    return LookaheadDecodingConfig(
        max_window_size=8,
        max_ngram_size=6,
        max_verification_set_size=8,
    )


# ### Build Configuration

# Finally, we'll specify the build configuration for the engine. This includes
# more typical parameters such as the max input length, the max number of tokens
# to be processing at once before queueing occurs, and the max number of prompts
# to process at once before queueing occurs:

ALLOW_CONCURRENT_INPUTS = 1


def get_build_config():
    from tensorrt_llm import BuildConfig

    return BuildConfig(
        plugin_config=get_plugin_config(),
        speculative_decoding_mode="LOOKAHEAD_DECODING",
        max_input_len=8192,
        max_num_tokens=16384,
        max_batch_size=ALLOW_CONCURRENT_INPUTS,
    )


# ## Serving inference under Dohertry's Threshold

# Now that we have setup everything necessary to compile the engine, we can setup up
# to serve it with Modal by creating an `App`.

app = modal.App("example-trtllm-inference-latency")

# Thanks to our custom container runtime system even this large, many gigabyte container boots in seconds.

# On the first container start, we mount the volume, download the model, and build the engine
# but subsequent starts will be much faster, as the engine is cached in the volume.
# Container starts are triggered when Modal scales up your infrastructure,
# like the first time you run this code or the first time a request comes in after a period of inactivity.

# Container lifecycles in Modal are managed via our `Cls` interface, so we define one below
# to manage the engine and run inference.
# For details, see [this guide](https://modal.com/docs/guide/lifecycle-functions).

MINUTES = 60  # seconds


@app.cls(
    image=tensorrt_image,
    allow_concurrent_inputs=ALLOW_CONCURRENT_INPUTS,
    gpu=GPU_CONFIG,
    scaledown_window=10 * MINUTES,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class Model:
    mode: str = modal.parameter(default="fast")

    def build_engine(self, engine_path, engine_kwargs) -> None:
        llm = LLM(model=self.model_path, **engine_kwargs)
        llm.save(engine_path)
        return llm

    @modal.enter()
    def enter(self):
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        self.model_path = MODELS_PATH / MODEL_ID

        print("downloading base model if necessary")
        snapshot_download(
            MODEL_ID,
            local_dir=self.model_path,
            ignore_patterns=["*.pt", "*.bin"],  # using safetensors
            revision=MODEL_REVISION,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        if self.mode == "fast":
            engine_kwargs = {
                "quant_config": get_quant_config(),
                "calib_config": get_calib_config(),
                "build_config": get_build_config(),
                "speculative_config": get_speculative_config(),
                "tensor_parallel_size": torch.cuda.device_count(),
            }
        else:
            engine_kwargs = {
                "tensor_parallel_size": torch.cuda.device_count(),
            }

        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,  # max generated tokens
            lookahead_config=engine_kwargs.get("speculative_config"),
        )

        engine_path = self.model_path / "trtllm_engine" / self.mode
        if not os.path.exists(engine_path):
            print(f"building new engine at {engine_path}")
            self.llm = self.build_engine(engine_path, engine_kwargs)
        else:
            print(f"loading engine from {engine_path}")
            self.llm = LLM(model=engine_path, **engine_kwargs)

    @modal.method()
    def generate(self, prompt) -> dict:
        SYSTEM_PROMPT = "You are a helpful, harmless, and honest AI assistant created by Meta."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        start_time = time.perf_counter()
        output = self.llm.generate(text, self.sampling_params)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return output.outputs[0].text, latency_ms

    @modal.method()
    def boot(self):
        pass

    @modal.exit()
    def shutdown(self):
        self.llm.shutdown()
        del self.llm


# ## Calling our inference function

# To run our `Model`'s `.generate` method from Python, we just need to call it --
# with `.remote` appended to run it on Modal.

# We wrap that logic in a `local_entrypoint` so you can run it from the command line with
# ```bash
# modal run trtllm_llama_latency.py
# ```

# which will output:

# ```
# mode=fast inference latency (p50, p90): (211.17ms, 883.27ms)
# ```

# If you want to see how slow the model is without all these optimizations, you can run:

# ```bash
# modal run trtllm_llama_latency.py --mode=slow
# ```

# which will output:

# ```
# mode=slow inference latency (p50, p90): (1140.88ms, 2274.24ms)
# ```

# For simplicity, we hard-code 10 questions to ask the model,
# and then run them one by one while recording the latency of each call.


@app.local_entrypoint()
def main(mode: str = "fast"):
    prompts = [
        "What atoms are in water?",
        "Which F1 team won in 2011?",
        "What is 12 * 9?",
        "Python function to print odd numbers between 1 and 10. Answer with code only.",
        "What is the capital of California?",
        "What's the tallest building in new york city?",
        "What year did the European Union form?",
        "How old was Geoff Hinton in 2022?",
        "Where is Berkeley?",
        "Are greyhounds or poodles faster?",
    ]

    print(f"creating container with mode={mode}")
    model = Model(mode=mode)

    print("cold booting container")
    model.boot.remote()

    print_queue = []
    latencies_ms = []
    for prompt in prompts:
        generated_text, latency_ms = model.generate.remote(prompt)

        print_queue.append((prompt, generated_text, latency_ms))
        latencies_ms.append(latency_ms)

    time.sleep(3)
    for prompt, generated_text, latency_ms in print_queue:
        print(f"Processed prompt in {latency_ms:.2f}ms")
        print(f"Prompt: {prompt}")
        print(f"Generated Text: {generated_text}")
        print("-" * 160)

    p50 = sorted(latencies_ms)[int(len(latencies_ms) * 0.5)]
    p90 = sorted(latencies_ms)[int(len(latencies_ms) * 0.9)]
    print(
        f"mode={mode} inference latency (p50, p90): ({p50:.2f}ms, {p90:.2f}ms)"
    )
