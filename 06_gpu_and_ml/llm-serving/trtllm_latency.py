# ---
# deploy: true
# ---
# # Serve an interactive language model app with latency-optimized TensorRT-LLM (LLaMA 3 8B)

# In this example, we demonstrate how to configure the TensorRT-LLM framework to serve
# Meta's LLaMA 3 8B model at interactive latencies on Modal.

# Many popular language model applications, like chatbots and code editing,
# put humans and models in direct interaction. According to an
# [oft-cited](https://lawsofux.com/doherty-threshold/)
# if [scientifically dubious](https://www.flashover.blog/posts/dohertys-threshold-is-a-lie)
# rule of thumb, computer systems need to keep their response times under 400ms
# in order to match pace with their human users.

# To hit this target, we use the [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
# inference framework from NVIDIA. TensorRT-LLM is the Lamborghini of inference engines:
# it achieves seriously impressive latency, but only if you tune it carefully.
# With the out-of-the-box defaults we observe an unacceptable median time
# to last token of over a second, but with careful configuration,
# we'll bring that down to under 250ms  -- over a 4x speed up!
# These latencies were measured on a single NVIDIA H100 GPU
# running LLaMA 3 8B on prompts and generations of a few dozen to a few hundred tokens.

# Here's what that looks like in a terminal chat interface:

# <video controls autoplay loop muted>
# <source src="https://modal-cdn.com/example-trtllm-latency.mp4" type="video/mp4">
# </video>

# ## Overview

# This guide is intended to document two things:

# 1. the [Python API](https://nvidia.github.io/TensorRT-LLM/llm-api)
# for building and running TensorRT-LLM engines, and

# 2. how to use recommendations from the
# [TensorRT-LLM performance guide](https://github.com/NVIDIA/TensorRT-LLM/blob/b763051ba429d60263949da95c701efe8acf7b9c/docs/source/performance/performance-tuning-guide/useful-build-time-flags.md)
# to optimize the engine for low latency.

# Be sure to check out TensorRT-LLM's
# [examples](https://nvidia.github.io/TensorRT-LLM/llm-api-examples)
# for sample code beyond what we cover here, like low-rank adapters (LoRAs).

# ### What is a TRT-LLM engine?

# The first step in running TensorRT-LLM is to build an "engine" from a model.
# Engines have a large number of parameters that must be tuned on a per-workload basis,
# so we carefully document the choices we made here and point you to additional resources
# that can help you optimize for your specific workload.

# Historically, this process was done with a clunky command-line-interface (CLI),
# but things have changed for the better!
# 2025 is [the year of CUDA Python](https://twitter.com/blelbach/status/1902842146232865280),
# including a new-and-improved Python SDK for TensorRT-LLM, supporting
# all the same features as the CLI -- quantization, speculative decoding, in-flight batching,
# and much more.

# ## Installing TensorRT-LLM

# To run TensorRT-LLM, we must first install it. Easier said than done!

# To run code on Modal, we define [container images](https://modal.com/docs/guide/images).
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.

# We start from an official `nvidia/cuda` container image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.

import time
from pathlib import Path

import modal

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.12",  # TRT-LLM requires Python 3.12
).entrypoint([])  # remove verbose logging by base image on entry

# On top of that, we add some system dependencies of TensorRT-LLM,
# including OpenMPI for distributed communication, some core software like `git`,
# and the `tensorrt_llm` package itself.

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt-llm==0.18.0",
    "pynvml<12",  # avoid breaking change to pynvml version API
    "flashinfer-python==0.2.5",
    "cuda-python==12.9.1",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

# Note that we're doing this by [method-chaining](https://quanticdev.com/articles/method-chaining/)
# a number of calls to methods on the `modal.Image`. If you're familiar with
# Dockerfiles, you can think of this as a Pythonic interface to instructions like `RUN` and `CMD`.

# End-to-end, this step takes about five minutes on first run.
# If you're reading this from top to bottom,
# you might want to stop here and execute the example
# with `modal run` so that it runs in the background while you read the rest.

# ## Downloading the model

# Next, we'll set up a few things to download the model to persistent storage and do it quickly --
# this is a latency-optimized example after all! For persistent, distributed storage, we use
# [Modal Volumes](https://modal.com/docs/guide/volumes), which can be accessed from any container
# with read speeds in excess of a gigabyte per second.

# We also set the `HF_HOME` environment variable to point to the Volume so that the model
# is cached there. And we install `hf-transfer` to get maximum download throughput from
# the Hugging Face Hub, in the hundreds of megabytes per second.

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
        "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",  # H100, silence noisy logs
    }
)

with tensorrt_image.imports():
    import os

    import torch
    from tensorrt_llm import LLM, SamplingParams

# ## Setting up the engine

# ### Quantization

# The amount of [GPU RAM](https://modal.com/gpu-glossary/device-hardware/gpu-ram)
# on a single card is a tight constraint for large models:
# RAM is measured in billions of bytes and large models have billions of parameters,
# each of which is two to four bytes.
# The performance cliff if you need to spill to CPU memory is steep,
# so all of those parameters must fit in the GPU memory,
# along with other things like the KV cache built up while processing prompts.

# The simplest way to reduce LLM inference's RAM requirements is to make the model's parameters smaller,
# fitting their values in a smaller number of bits, like four or eight. This is known as _quantization_.

# NVIDIA's [Ada Lovelace/Hopper chips](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
# like the L40S and H100, are capable of native 8bit floating point calculations
# in their [Tensor Cores](https://modal.com/gpu-glossary/device-hardware/tensor-core),
# so we choose that as our quantization format.
# These GPUs are capable of twice as many floating point operations per second in 8bit as in 16bit --
# about two quadrillion per second on an H100 SXM.

# Quantization buys us two things:

# - faster startup, since less data has to be moved over the network onto CPU and GPU RAM

# - faster inference, since we get twice the FLOP/s and less data has to be moved from GPU RAM into
# [on-chip memory](https://modal.com/gpu-glossary/device-hardware/l1-data-cache) and
# [registers](https://modal.com/gpu-glossary/device-hardware/register-file)
# with each computation

# We'll use TensorRT-LLM's `QuantConfig` to specify that we want `FP8` quantization.
# [See their code](https://github.com/NVIDIA/TensorRT-LLM/blob/88e1c90fd0484de061ecfbacfc78a4a8900a4ace/tensorrt_llm/models/modeling_utils.py#L184)
# for more options.

N_GPUS = 1  # Bumping this to 2 will improve latencies further but not 2x
GPU_CONFIG = f"H100:{N_GPUS}"


def get_quant_config():
    from tensorrt_llm.llmapi import QuantConfig

    return QuantConfig(quant_algo="FP8")


# Quantization is a lossy compression technique. The impact on model quality can be
# minimized by tuning the quantization parameters on even a small dataset. Typically, we
# see less than 2% degradation in evaluation metrics when using `fp8`. We'll use the
# `CalibrationConfig` class to specify the calibration dataset.


def get_calib_config():
    from tensorrt_llm.llmapi import CalibConfig

    return CalibConfig(
        calib_batches=512,
        calib_batch_size=1,
        calib_max_seq_length=2048,
        tokenizer_max_seq_length=4096,
    )


# ### Configure plugins

# TensorRT-LLM is an LLM inference framework built on top of NVIDIA's TensorRT,
# which is a generic inference framework for neural networks.

# TensorRT includes a "plugin" extension system that allows you to adjust behavior,
# like configuring the [CUDA kernels](https://modal.com/gpu-glossary/device-software/kernel)
# used by the engine.
# The [General Matrix Multiply (GEMM)](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
# plugin, for instance, adds heavily-optimized matrix multiplication kernels
# from NVIDIA's [cuBLAS library of linear algebra routines](https://docs.nvidia.com/cuda/cublas/).

# We'll specify a number of plugins for our engine implementation.
# The first is
# [multiple profiles](https://github.com/NVIDIA/TensorRT-LLM/blob/b763051ba429d60263949da95c701efe8acf7b9c/docs/source/performance/performance-tuning-guide/useful-build-time-flags.md#multiple-profiles),
# which configures TensorRT to prepare multiple kernels for each high-level operation,
# where different kernels are optimized for different input sizes.
# The second is `paged_kv_cache` which enables a
# [paged attention algorithm](https://arxiv.org/abs/2309.06180)
# for the key-value (KV) cache.

# The last two parameters are GEMM plugins optimized specifically for low latency,
# rather than the more typical high arithmetic throughput,
# the `low_latency` plugins for `gemm` and `gemm_swiglu`.

# The `low_latency_gemm_swiglu_plugin` plugin fuses the two matmul operations
# and non-linearity of the feedforward component of the Transformer block into a single kernel,
# reducing round trips between GPU
# [cache memory](https://modal.com/gpu-glossary/device-hardware/l1-data-cache)
# and RAM. For details on kernel fusion, see
# [this blog post by Horace He of Thinking Machines](https://horace.io/brrr_intro.html).
# Note that at the time of writing, this only works for `FP8` on Hopper GPUs.

# The `low_latency_gemm_plugin` is a variant of the GEMM plugin that brings in latency-optimized
# kernels from NVIDIA's [CUTLASS library](https://github.com/NVIDIA/cutlass).


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


# ### Configure speculative decoding

# Speculative decoding is a technique for generating multiple tokens per step,
# avoiding the auto-regressive bottleneck in the Transformer architecture.
# Generating multiple tokens in parallel exposes more parallelism to the GPU.
# It works best for text that has predicable patterns, like code,
# but it's worth testing for any workload where latency is critical.

# Speculative decoding can use any technique to guess tokens, including running another,
# smaller language model. Here, we'll use a simple, but popular and effective
# speculative decoding strategy called "lookahead decoding",
# which essentially guesses that token sequences from the past will occur again.


def get_speculative_config():
    from tensorrt_llm.llmapi import LookaheadDecodingConfig

    return LookaheadDecodingConfig(
        max_window_size=8,
        max_ngram_size=6,
        max_verification_set_size=8,
    )


# ### Set the build config

# Finally, we'll specify the overall build configuration for the engine. This includes
# more obvious parameters such as the maximum input length, the maximum number of tokens
# to process at once before queueing occurs, and the maximum number of sequences
# to process at once before queueing occurs.

# To minimize latency, we set the maximum number of sequences (the "batch size")
# to just one. We enforce this maximum by setting the number of inputs that the
# Modal Function is allowed to process at once -- `max_concurrent_inputs`.
# The default is `1`, so we don't need to set it, but we are setting it explicitly
# here in case you want to run this code with a different balance of latency and throughput.

MAX_BATCH_SIZE = MAX_CONCURRENT_INPUTS = 1


def get_build_config():
    from tensorrt_llm import BuildConfig

    return BuildConfig(
        plugin_config=get_plugin_config(),
        speculative_decoding_mode="LOOKAHEAD_DECODING",
        max_input_len=8192,
        max_num_tokens=16384,
        max_batch_size=MAX_BATCH_SIZE,
    )


# ## Serving inference under the Doherty Threshold

# Now that we have written the code to compile the engine, we can
# serve it with Modal!

# We start by creating an `App`.

app = modal.App("example-trtllm-latency")

# Thanks to our [custom container runtime system](https://modal.com/blog/jono-containers-talk),
# even this large container boots in seconds.

# On the first container start, we mount the Volume, download the model, and build the engine,
# which takes a few minutes. Subsequent starts will be much faster,
# as the engine is cached in the Volume and loaded in seconds.

# Container starts are triggered when Modal scales up your Function,
# like the first time you run this code or the first time a request comes in after a period of inactivity.
# For details on optimizing container start latency, see
# [this guide](https://modal.com/docs/guide/cold-start).

# Container lifecycles in Modal are managed via our `Cls` interface, so we define one below
# to separate out the engine startup (`enter`) and engine execution (`generate`).
# For details, see [this guide](https://modal.com/docs/guide/lifecycle-functions).

MINUTES = 60  # seconds


@app.cls(
    image=tensorrt_image,
    gpu=GPU_CONFIG,
    scaledown_window=10 * MINUTES,
    timeout=10 * MINUTES,
    volumes={VOLUME_PATH: volume},
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
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
        start_time = time.perf_counter()
        text = self.text_from_prompt(prompt)
        output = self.llm.generate(text, self.sampling_params)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return output.outputs[0].text, latency_ms

    @modal.method()
    async def generate_async(self, prompt):
        text = self.text_from_prompt(prompt)
        async for output in self.llm.generate_async(
            text, self.sampling_params, streaming=True
        ):
            yield output.outputs[0].text_diff

    def text_from_prompt(self, prompt):
        SYSTEM_PROMPT = (
            "You are a helpful, harmless, and honest AI assistant created by Meta."
        )

        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + prompt

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @modal.method()
    def boot(self):
        pass  # no-op to start up containers

    @modal.exit()
    def shutdown(self):
        self.llm.shutdown()
        del self.llm


# ## Calling our inference function

# To run our `Model`'s `.generate` method from Python, we just need to call it --
# with `.remote` appended to run it on Modal.

# We wrap that logic in a `local_entrypoint` so you can run it from the command line with

# ```bash
# modal run trtllm_latency.py
# ```

# which will output something like:

# ```
# mode=fast inference latency (p50, p90): (211.17ms, 883.27ms)
# ```

# Use `--mode=slow` to see model latency without optimizations.

# ```bash
# modal run trtllm_latency.py --mode=slow
# ```

# which will output something like

# ```
# mode=slow inference latency (p50, p90): (1140.88ms, 2274.24ms)
# ```

# For simplicity, we hard-code 10 questions to ask the model,
# then run them one by one while recording the latency of each call.
# But the code in the `local_entrypoint` is just regular Python code
# that runs on your machine -- we wrap it in a CLI automatically --
# so feel free to customize it to your liking.


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

    print(f"🏎️  creating container with mode={mode}")
    model = Model(mode=mode)

    print("🏎️  cold booting container")
    model.boot.remote()

    print_queue = []
    latencies_ms = []
    for prompt in prompts:
        generated_text, latency_ms = model.generate.remote(prompt)

        print_queue.append((prompt, generated_text, latency_ms))
        latencies_ms.append(latency_ms)

    time.sleep(3)  # allow remote prints to clear
    for prompt, generated_text, latency_ms in print_queue:
        print(f"Processed prompt in {latency_ms:.2f}ms")
        print(f"Prompt: {prompt}")
        print(f"Generated Text: {generated_text}")
        print("🏎️ " * 20)

    p50 = sorted(latencies_ms)[int(len(latencies_ms) * 0.5) - 1]
    p90 = sorted(latencies_ms)[int(len(latencies_ms) * 0.9) - 1]
    print(f"🏎️  mode={mode} inference latency (p50, p90): ({p50:.2f}ms, {p90:.2f}ms)")


# Once deployed with `modal deploy`, this `Model.generate` function
# can be called from other Python code. It can also be converted to an HTTP endpoint
# for invocation over the Internet by any client.
# For details, see [this guide](https://modal.com/docs/guide/trigger-deployed-functions).

# As a quick demo, we've included some sample chat client code in the
# Python main entrypoint below. To use it, first deploy with

# ```bash
# modal deploy trtllm_latency.py
# ```

# and then run the client with

# ```python notest
# python trtllm_latency.py
# ```


if __name__ == "__main__":
    import sys

    try:
        Model = modal.Cls.from_name("example-trtllm-latency", "Model")
        print("🏎️  connecting to model")
        model = Model(mode=sys.argv[1] if len(sys.argv) > 1 else "fast")
        model.boot.remote()
    except modal.exception.NotFoundError as e:
        raise SystemError("Deploy this app first with modal deploy") from e

    print("🏎️  starting chat. exit with :q, ctrl+C, or ctrl+D")
    try:
        prompt = []
        while (nxt := input("🏎️  > ")) != ":q":
            prompt.append({"role": "user", "content": nxt})
            resp = ""
            for out in model.generate_async.remote_gen(prompt):
                print(out, end="", flush=True)
                resp += out
            print("\n")
            prompt.append({"role": "assistant", "content": resp})
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    finally:
        print("\n")
        sys.exit(0)
