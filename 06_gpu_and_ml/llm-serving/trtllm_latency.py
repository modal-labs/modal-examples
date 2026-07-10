# # Serve an interactive language model app with low-latency TensorRT-LLM (LLaMA 3 8B)

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
# We pair it with [Modal Servers](https://modal.com/docs/guide/servers) which routes requests
# through a [new, low-latency proxy service](https://modal.com/blog/serverless-servers)
# designed for latency-sensitive inference workloads,
# minimizing the overhead between client and GPU.
# These latencies were measured on a single NVIDIA H100 GPU
# running LLaMA 3 8B on prompts and generations of a few dozen to a few hundred tokens.

# Here's what that looks like in a terminal chat interface:

# <video controls autoplay loop muted>
# <source src="https://modal-cdn.com/example-trtllm-latency.mp4" type="video/mp4">
# </video>

# ## Overview

# This guide documents how to use recommendations from the
# [TensorRT-LLM performance guide](https://github.com/NVIDIA/TensorRT-LLM/blob/b763051ba429d60263949da95c701efe8acf7b9c/docs/source/performance/performance-tuning-guide/useful-build-time-flags.md)
# to optimize a [TensorRT-LLM engine](https://nvidia.github.io/TensorRT-LLM/llm-api) for low latency,
# then serve it behind an OpenAI-compatible HTTP API with `trtllm-serve`.

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
# There is now a new-and-improved Python SDK for TensorRT-LLM, supporting
# all the same features as the CLI -- quantization, speculative decoding, in-flight batching,
# and much more.

# ## Set up the container image

# To run code on Modal, we define [container images](https://modal.com/docs/guide/images).
# All Modal containers have access to GPU drivers via the underlying host environment,
# but we still need to install the software stack on top of the drivers, from the CUDA runtime up.

# We start from an official `nvidia/cuda` container image,
# which includes the CUDA runtime & development libraries
# and the environment configuration necessary to run them.
# On top of that, we add some system dependencies of TensorRT-LLM,
# including OpenMPI for distributed communication, some core software like `git`,
# and the `tensorrt_llm` package itself.

# While we're at it, we import the dependencies we'll need both remotely and locally (for deployment).

import asyncio
import json
import subprocess
import time
from pathlib import Path

import aiohttp
import modal
import modal.experimental

MINUTES = 60  # seconds

tensorrt_image = modal.Image.from_registry(
    "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    add_python="3.12",  # TRT-LLM requires Python 3.12
).entrypoint([])  # silence noisy NVIDIA license logging

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).uv_pip_install(
    "tensorrt-llm==0.20.0",  # 0.20+ adds trtllm-serve --extra_llm_api_options
    "pynvml>=12",  # required by tensorrt-llm 0.20
    "flashinfer-python==0.2.5",
    "cuda-python==12.9.1",
    "onnx==1.19.1",
    "mpmath==1.3.0",
    "torch==2.7.0",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

# Note that we're doing this by [method-chaining](https://quanticdev.com/articles/method-chaining/)
# a number of calls to methods on the `modal.Image`. If you're familiar with
# Dockerfiles, you can think of this as a Pythonic interface to instructions like `RUN` and `CMD`.

# End-to-end, this step takes about five minutes on first run.

# ## Cache model weights in a Modal Volume

# We serve [Meta's LLaMA 3 8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct),
# downloading it to persistent storage and loading it quickly --
# this is a latency-optimized example after all! For persistent, distributed storage, we use
# [Modal Volumes](https://modal.com/docs/guide/volumes), which can be accessed from any container
# with read speeds in excess of a gigabyte per second.

# We also set the `HF_HOME` environment variable to point to the Volume so that the model
# is cached there, and turn on
# [high-performance downloads](https://huggingface.co/docs/hub/en/models-downloading#faster-downloads)
# to get maximum throughput from the Hugging Face Hub.

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"  # fork without repo gating
MODEL_REVISION = "53346005fb0ef11d3b6a83b12c895cca40156b6c"  # pin to avoid surprises!

volume = modal.Volume.from_name(
    "example-trtllm-inference-volume", create_if_missing=True
)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

tensorrt_image = tensorrt_image.uv_pip_install(
    "huggingface_hub==0.36.0",
).env(
    {
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HOME": str(MODELS_PATH),
        "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",  # H100, silence noisy logs
    }
)

# ## Configure for low latency

# ### Quantization

# The amount of [GPU RAM](https://modal.com/gpu-glossary/device-hardware/gpu-ram)
# on a single card is a tight constraint for large models:
# RAM is measured in billions of bytes and large models have billions of parameters,
# each of which is two to four bytes.
# The performance cliff if you need to spill to CPU memory is steep,
# so all of those parameters must fit in the GPU memory,
# along with other things like the KV cache built up while processing prompts.

# The simplest way to reduce LLM inference's RAM requirements is to make the model's parameters smaller,
# fitting their values in a smaller number of bits, like four or eight. This is known as
# [_quantization_](https://modal.com/llm-almanac/quant-formats).

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

# Quantization is a lossy compression technique. The impact on model quality can be
# minimized by tuning the quantization parameters on even a small dataset. Typically, we
# see less than 2% degradation in evaluation metrics when using `fp8`. We use the
# `CalibConfig` class to specify the calibration dataset.

# ### Configure plugins

# TensorRT-LLM is an LLM inference framework built on top of NVIDIA's TensorRT,
# which is a generic inference framework for neural networks.

# TensorRT includes a "plugin" extension system that allows you to adjust behavior,
# like configuring the [CUDA kernels](https://modal.com/gpu-glossary/device-software/kernel)
# used by the engine.
# The [General Matrix Multiply (GEMM)](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
# plugin, for instance, adds heavily-optimized matrix multiplication kernels
# from NVIDIA's [cuBLAS library of linear algebra routines](https://docs.nvidia.com/cuda/cublas/).

# We specify a number of plugins for our engine implementation.
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

# ### A note on speculative decoding

# Speculative decoding is a technique for generating multiple tokens per step,
# avoiding the auto-regressive bottleneck in the Transformer architecture and
# exposing more parallelism to the GPU. It works best for text with predictable
# patterns, like code, but it's worth testing for any latency-critical workload.
# We no longer configure it by hand here -- it's handled within the serving stack --
# but it remains one of the most effective levers for cutting per-token latency.

# ### Set the build config

# Finally, we specify the overall build configuration for the engine. This includes
# the maximum input length, the maximum number of tokens
# to process at once before queueing occurs, and the maximum number of sequences
# to process at once before queueing occurs.

# To minimize latency, we set the maximum number of sequences (the "batch size")
# to just one and pair that with low per-container concurrency below,
# trading throughput for the lowest possible per-request latency.

N_GPUS = 1  # bumping this to 2 will improve latencies further but not 2x
GPU = f"H100:{N_GPUS}"
MAX_BATCH_SIZE = 1  # minimize latency by processing one request at a time

# ## Define the inference server and infrastructure

# ### Selecting infrastructure to minimize latency

# Minimizing latency requires geographic co-location of clients and servers.

# So for low latency LLM inference services on Modal, you must select a
# [cloud region](https://modal.com/docs/guide/region-selection)
# for both the GPU-accelerated containers running inference
# and for the [internal Modal proxy system](https://modal.com/blog/serverless-servers)
# that forwards requests to them as part of defining a Server.

# Here, we assume users are mostly in the northern half of the Americas
# and select the `us` cloud region with a nearby `us-west` proxy to serve them.
# This should result in at most a few dozen milliseconds of round-trip time.

REGION = "us"
PORT = 8000
PROXY_REGION = "us-west"

# Latencies for multi-turn interactions with LLMs are
# substantially cut when previous interaction turns are in the KV cache.
# KV caches are stored in [GPU RAM](https://modal.com/gpu-glossary/device-hardware/gpu-ram),
# so they aren't shared across replicas.
# To improve cache hit rate, Modal Servers
# includes sticky routing based on a client-provided header.
# See [this code sample](https://modal.com/docs/examples/server_sticky)
# for details.

# For production-scale LLM inference services, there are generally
# enough requests to justify keeping at least one replica running at all times.
# Having a "warm" or "live" replica reduces latency by skipping slow initialization work
# that occurs when a new replica boots up (a ["cold start"](https://modal.com/docs/guide/cold-start)).
# For LLM inference servers, that latency runs from seconds to minutes.

# To ensure at least one container is always available,
# we can set the `min_containers` of our Modal Function to `1` or more.
# However, since this is documentation code, we'll set it to `0`
# to avoid surprise bills during casual use.

MIN_CONTAINERS = 0  # set to 1 to ensure one replica is always ready

# Finally, we set a target for the number of inputs to run on a single container
# with [`modal.concurrent`](https://modal.com/docs/reference/modal.concurrent).
# For details, see [the guide](https://modal.com/docs/guide/concurrent-inputs).
# We keep concurrency low for minimum per-request latency.

TARGET_INPUTS = 1  # low concurrency for minimum per-request latency

# ### Health check and warmup helpers

# Modal considers a new replica ready to receive inputs once the `modal.enter` methods have exited
# and the container accepts connections.
# To ensure that we actually finish setting up our server before we are marked ready for inputs,
# we poll the server's `/health` endpoint until it's ready,
# then send a few warm-up requests so the first real request isn't slow.

# We use the [`requests` library](https://requests.readthedocs.io/en/latest/)
# to send ourselves these HTTP requests on
# [`localhost`/`127.0.0.1`](https://superuser.com/questions/31824/why-is-localhost-ip-127-0-0-1).

with tensorrt_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = 20 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(5)
    raise TimeoutError(f"TensorRT-LLM server not ready within {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def warmup():
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload,
            timeout=60,
        ).raise_for_status()


# ### Build the TRT-LLM engine and start the server

# We use [`modal.enter/exit`](https://modal.com/docs/guide/lifecycle-functions) to manage
# the server lifecycle. On the first container start, we build an optimized engine
# using the TensorRT-LLM [Python API](https://nvidia.github.io/TensorRT-LLM/llm-api)
# with FP8 quantization and low-latency plugins, then cache it in the Volume.
# Subsequent starts load the cached engine in seconds and launch `trtllm-serve`
# to expose an OpenAI-compatible HTTP API.

# The key decorators are [`@app.server`](https://modal.com/docs/guide/servers),
# [`@modal.enter`, and `@modal.exit`](https://modal.com/docs/guide/lifecycle-functions)
# The code in the `enter` decorator needs to start a server process that listens on a port.

# The `@app.server` decorator does a lot! We:

# 1. Attach our Image
# 2. Request a GPU
# 3. Attach our cache Volume
# 4. Specify the regions for the routing proxy and compute
# 5. Configure auto-scaling, concurrency, and timeouts
# 6. Configure authentication via [Proxy Tokens](https://modal.com/docs/guide/webhook-proxy-auth) (disabled here for demo purposes)

app = modal.App("example-trtllm-low-latency")


@app.server(
    image=tensorrt_image,
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    compute_region=REGION,
    routing_region=PROXY_REGION,  # location of proxies, should be close to Cls region
    min_containers=MIN_CONTAINERS,
    target_concurrency=TARGET_INPUTS,
    startup_timeout=20 * MINUTES,
    port=PORT,  # wrapped code must listen on this port
    exit_grace_period=15,  # seconds, time to finish up requests when closing down
    unauthenticated=True,  # no auth for this demo, see Servers guide/docs for auth details
)
class TRT:
    @modal.enter()
    def startup(self):
        """Download model, build/load the optimized engine, and start trtllm-serve."""
        from huggingface_hub import snapshot_download
        from tensorrt_llm import LLM, BuildConfig
        from tensorrt_llm.llmapi import CalibConfig, QuantConfig
        from tensorrt_llm.plugin.plugin import PluginConfig

        model_path = str(MODELS_PATH / MODEL_ID)
        engine_path = str(MODELS_PATH / MODEL_ID / "trtllm_engine" / "serve-0.20")

        snapshot_download(
            MODEL_ID,
            local_dir=model_path,
            ignore_patterns=["*.pt", "*.bin"],  # using safetensors
            revision=MODEL_REVISION,
        )

        if not Path(engine_path).exists():
            print(f"building new engine at {engine_path}")
            llm = LLM(
                model=model_path,
                quant_config=QuantConfig(quant_algo="FP8"),
                calib_config=CalibConfig(
                    calib_batches=512,
                    calib_batch_size=1,
                    calib_max_seq_length=2048,
                    tokenizer_max_seq_length=4096,
                ),
                build_config=BuildConfig(
                    plugin_config=PluginConfig.from_dict(
                        {
                            "multiple_profiles": True,
                            "paged_kv_cache": True,
                            "low_latency_gemm_swiglu_plugin": "fp8",
                            "low_latency_gemm_plugin": "fp8",
                        }
                    ),
                    max_input_len=8192,
                    max_num_tokens=16384,
                    max_batch_size=MAX_BATCH_SIZE,
                ),
                tensor_parallel_size=N_GPUS,
            )
            llm.save(engine_path)
            llm.shutdown()
            del llm
            volume.commit()
        else:
            print(f"loading cached engine from {engine_path}")

        # When serving a prebuilt TensorRT engine, `trtllm-serve` needs to be
        # pointed at the original Hugging Face checkpoint for the tokenizer and
        # its chat template, otherwise `/v1/chat/completions` returns a 400.

        cmd = [
            "trtllm-serve",
            engine_path,
            "--tokenizer",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
        ]

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Deploy the server

# To deploy the server on Modal, just run

# ```bash
# modal deploy trtllm_latency.py
# ```

# This will create a new App on Modal and build the container image for it if it hasn't been built yet.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-trtllm-low-latency-trt.us-west.modal.direct`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands.
# For simple routes, you can even send a request directly from the docs page.

# Note: when no replicas are available, Modal will respond with
# the [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503).
# In your browser, you can just hit refresh until the docs page appears.
# You can see the status of the application and its containers on your [Modal dashboard](https://modal.com/apps).

# ## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that hits the server with a simple client.

# If you execute the command

# ```bash
# modal run trtllm_latency.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, prompt=None, twice=True):
    url = await TRT.get_url.aio()

    system_prompt = {
        "role": "system",
        "content": "You are a helpful, harmless, and honest AI assistant.",
    }
    if prompt is None:
        prompt = "What is the capital of France?"

    content = [{"type": "text", "text": prompt}]

    messages = [
        system_prompt,
        {"role": "user", "content": content},
    ]

    await probe(url, messages, timeout=test_timeout)
    if twice:
        messages[0]["content"] = "You are a pirate."
        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await probe(url, messages, timeout=test_timeout)


# This test relies on the two helper functions below,
# which ping the server and wait for a valid response to stream.

# The `probe` helper function specifically ignores
# two types of errors that can occur while a replica
# is starting up -- timeouts on the client and 5XX responses from the server.
# Modal Servers returns the [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)
# when there are no live replicas.

# We include a header with each request -- `Modal-Session-ID`.
# The value associated with this key
# is used to map requests onto containers such that
# while the set of containers is fixed, requests with the same value
# are sent to the same container.
# Set this to a different value per multi-turn interaction
# (prototypically, a user conversation thread with a chatbot)
# to improve KV cache hit rates.
# Note that this header is only compatible with Modal Servers.


async def probe(url, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [{"role": "user", "content": "Tell me a joke."}]

    client_id = str(0)  # set per multi-turn interaction for sticky routing
    headers = {"Modal-Session-ID": client_id}
    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        while time.time() < deadline:
            try:
                await _send_request_streaming(session, messages)
                return
            except asyncio.TimeoutError:
                await asyncio.sleep(1)
            except aiohttp.client_exceptions.ClientResponseError as e:
                if e.status == 503:
                    await asyncio.sleep(1)
                    continue
                raise e
    raise TimeoutError(f"No response from server within {timeout} seconds")


async def _send_request_streaming(
    session: aiohttp.ClientSession, messages: list, timeout: int | None = None
) -> None:
    payload = {"model": MODEL_ID, "messages": messages, "stream": True}
    headers = {"Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        full_text = ""

        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Server-Sent Events format: "data: ...."
            if not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                # ignore any non-JSON keepalive
                continue

            delta = (evt.get("choices") or [{}])[0].get("delta") or {}
            chunk = delta.get("content")

            if chunk:
                print(chunk, end="", flush="\n" in chunk or "." in chunk)
                full_text += chunk
        print()  # newline after stream completes
