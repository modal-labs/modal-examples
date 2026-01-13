# # Low latency Qwen 3 8B with SGLang and Modal

# In this example, we show how to serve [SGLang](https://github.com/sgl-project/sglang) at low latency on Modal.

# This example is intended to demonstrate everything required to run
# inference at the highest performance and with the lowest latency possible,
# and so it includes advanced features of both SGLang and Modal.
# For a simpler introduction to LLM serving, see
# [this example](https://modal.com/docs/examples/llm_inference).

# To minimize routing overheads, we use `@modal.experimental.http_server`,
# which uses a new, low-latency routing service on Modal designed for latency-sensitive inference workloads.
# This gives us more control over routing, but with increased power comes increased responsibility.

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/images).

# We start from a container image provided
# [by the SGLang team via Dockerhub](https://hub.docker.com/r/lmsysorg/sglang/tags).

# While we're at it, we import the dependencies we'll need both remotely and locally (for deployment).

import asyncio
import json
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60  # seconds

sglang_image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:v0.5.6.post2-cu129-amd64-runtime"
    ).entrypoint([])  # silence chatty logs on container start
)

# We also choose a [GPU](https://modal.com/docs/guide/gpu) to deploy our inference server onto.
# We choose the [H100 GPU](https://modal.com/blog/introducing-h100),
# which offers excellent price-performance
# and supports 8bit floating point operations, which are the
# lowest precision well-supported in the relevant [GPU kernels](https://modal.com/gpu-glossary/device-software/kernel)
# across a variety of model architectures.

# Below, we discuss the choice of GPU count.

GPU_TYPE, N_GPUS = "H100!", 2
GPU = f"{GPU_TYPE}:{N_GPUS}"

# ### Loading and cacheing the model weights

# We'll serve [Alibaba's Qwen 3 LLM](https://www.alibabacloud.com/blog/alibaba-introduces-qwen3-setting-new-benchmark-in-open-source-ai-with-hybrid-reasoning_602192).
# For lower latency, we pick a smaller model (8B params)
# in a lower precision floating point format (FP8).
# This reduces the amount of data that needs to be loaded
# [from GPU RAM into SM SRAM](https://modal.com/gpu-glossary/perf/memory-bandwidth)
# in each forward pass.

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
MODEL_REVISION = (
    "220b46e3b2180893580a4454f21f22d3ebb187d3"  # latest commit as of 2026-01-01
)

# We load the model [from the Hugging Face Hub](https://huggingface.co/collections/Qwen/qwen3),
# so we'll need their Python package.

sglang_image = sglang_image.uv_pip_install("huggingface-hub==0.36.0")

# We don't want to load the model from the Hub every time we start the server.
# We can load it much faster from a [Modal Volume](https://modal.com/docs/guide/volumes).
# Typical speeds are around one to two GB/s.

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
MODEL_PATH = f"{HF_CACHE_PATH}/{MODEL_NAME}"

# In addition to pointing the Hugging Face Hub at the path
# where we mount the Volume, we also
# [turn on "high performance" downloads](https://huggingface.co/docs/hub/en/models-downloading#faster-downloads),
# which can fully saturate our network bandwidth.

sglang_image = sglang_image.env(
    {"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"}
)

# ### Cacheing compilation artifacts

# Model weights aren't the only thing we want to cache.

# As a rule, LLM inference servers like SGLang don't directly provide their own kernels.
# They draw high-performance kernels from a variety of sources.

# As of version `0.5.6`, SGLang's default kernel backend
# for FP8 matrix multiplications (`fp8-gemm-backend`)
# on Hopper [SM architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
# GPUs like the H100 is
# [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
# by DeepSeek.

# The binaries of these kernels are not included in the SGLang Docker image and so
# must be [JIT-compiled](https://modal.com/gpu-glossary/host-software/nvrtc).
# We store these in a Modal Volume as well.

DG_CACHE_VOL = modal.Volume.from_name("deepgemm-cache", create_if_missing=True)
DG_CACHE_PATH = "/root/.cache/deepgemm"

# JIT DeepGEMM kernels are on by default, but we explicitly enable them via an environment variable.

sglang_image = sglang_image.env({"SGLANG_ENABLE_JIT_DEEPGEMM": "1"})

# We trigger the compilation by running `sglang.compile_deep_gemm` in a `subprocess`
# kicked off from a Python function.


def compile_deep_gemm():
    import os

    if int(os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1")):
        subprocess.run(
            f"python3 -m sglang.compile_deep_gemm --model-path {MODEL_NAME} --revision {MODEL_REVISION} --tp {N_GPUS}",
            shell=True,
        )


# We run this Python function on Modal as part of building the Image
# so that it has access to the appropriate GPU and the caches for our model and compilaton artifacts.

sglang_image = sglang_image.run_function(
    compile_deep_gemm,
    volumes={DG_CACHE_PATH: DG_CACHE_VOL, HF_CACHE_PATH: HF_CACHE_VOL},
    gpu=GPU,
)

# ## Configure SGLang for minimal latency

# LLM inference engines like SGLang come with a wide variety of "knobs" to tune performance.

# To determine the appropriate configuration to hit latency and throughput service objectives,
# we recommend [application-specific benchmarking](https://modal.com/llm-almanac/how-to-benchmark)
# guided by [published generic benchmarks](https://modal.com/llm-almanac/advisor).

# Here, we assume that the primary goal is to minimize per-request latency, with less regard to throughput
# (and so to cost) and walk through some of the key choices.

# The primary contributor to per-request latency is the time to move all of the model's weights (multiple gigabytes)
# from [GPU RAM](https://modal.com/gpu-glossary/device-hardware/gpu-ram)
# into [SRAM in the Streaming Multiprocessors](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor),
# which must be done at least once in the course of processing a request --
# naively, once per token per request.
# The time taken is limited by the
# [memory bandwidth](https://modal.com/gpu-glossary/perf/memory-bandwidth)
# between those two stores, which is on the order of terabytes per second on modern data center GPUs.
# With models at the scale of gigabytes, a token will take milliseconds to generate --
# or whole seconds for the kilotoken responses users are accustomed to.

# We use two strategies to cut latency in our [memory-bound](https://modal.com/gpu-glossary/perf/memory-bound) workload:

# - operate across multiple GPUs for more aggregate bandwidth and faster loads, with tensor parallelism

# - generate more tokens per load, with speculative decoding

# ### Increasing effective memory bandwidth with tensor parallelism

# Running SGLang on two H100s will double our effective
# [memory bandwidth](https://modal.com/gpu-glossary/perf/memory-bandwidth)
# during large matrix multiplications.


# Matrices are also known as tensors, and so this strategy that takes advantage
# of the inherent parallelism within matrix multiplication is known as _tensor parallelism_.

# Actual speedups are generally less than what you get from "napkin math" based on available bandwidths --
# we observed a speedup of about 30% moving from one to two H100s when developing this example, rather than 100%.

# ### Parallelizing token generation with speculative decoding

# Transformer and recurrent language models generate text sequentially:
# the model's output at step `i` is part of the input at step `i+1`.
# Per Amdahl's Law, that sequential work becomes the bottleneck
# as other steps get faster from increased parallelism.

# The solution is to generate more tokens on each step.
# The primary technique to do so without changing model behavior is known as
# [_speculative decoding_](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/),
# which "speculates" a number of draft tokens and verifies them in parallel with the primary model.

# Speculative decoding techniques themselves have a number of parameters, the most important
# of which is the technique to use to generate draft tokens.
# Simple techniques based on n-grams are a good place to start.
# But in our experience, the [EAGLE-3](https://arxiv.org/abs/2503.01840)
# technique gives enough of a performance boost to be worth
# the overhead of maintaining an extra model for speculation.

# And for popular models, you can often find a high-quality EAGLE-3 draft model
# with open weights. For Qwen 3-8B, we like
# [`Tengyunw`'s model](https://huggingface.co/Tengyunw/qwen3_8b_eagle3).

speculative_config = {
    "speculative-algorithm": "EAGLE3",
    "speculative-draft-model-path": "Tengyunw/qwen3_8b_eagle3",
}

# We adopt the default configuration for this model from the documentation.
# With these settings, we observed an ~30% boost in throughput for a single user
# during the development of this sample code.

speculative_config |= {
    "speculative-num-steps": 6,
    "speculative-eagle-topk": 10,
    "speculative-num-draft-tokens": 32,
}

# Note that unlike tensor parallelism,
# speculative decoding is not good for
# [compute-bound](https://modal.com/gpu-glossary/perf/compute-bound)
# workloads, since it generally increases demand for
# [arithmetic bandwidth](https://modal.com/gpu-glossary/perf/arithmetic-bandwidth).
# So for workloads that admit larger batch sizes for requests,
# on the scale of dozens to hundreds, speculative decoding is not recommended.

# ## Define the inference server and infrastructure

# ### Selecting infrastructure to minimize latency

# Minimizing latency requires geographic co-location of clients and servers.

# So for low latency LLM inference services on Modal, you must select a
# [cloud region](https://modal.com/docs/guide/region-selection)
# for both the GPU-accelerated containers running inference
# and for the internal Modal proxies that forward requests to them
# as part of defining a `modal.experimental.http_server`.

# Here, we assume users are mostly in the northern half of the Americas
# and select the `us-east` cloud region serve them.
# This should result in at most a few dozen milliseconds of round-trip time.

REGION = "us-east"

# For production-scale LLM inference services, there are generally
# enough requests to justify keeping at least one replica running at all times.
# Having a "warm" or "live" replica reduces latency by skipping slow initialization work
# that occurs when new replica boots up (a ["cold start"](https://modal.com/docs/guide/cold-start)).
# For LLM inference servers, that latency runs from seconds to minutes.

# To ensure at least one container is always available,
# we can set the `min_containers` of our Modal Function
# to `1` or more.

# However, since this is documentation code, we'll set it to `0`
# to avoid surprise bills during casual use.

MIN_CONTAINERS = 0  # set to 1 to ensure one replica is always ready

# Finally, we need to decide how we will scale up and down replicas
# in response to load. Without autoscaling, users' requests will queue
# when the server becomes overloaded. Even apart from queueing, responses
# generally become slower per user above a certain minimum number of
# concurrent requests.

# So we set a target for the number of inputs to run on a single container
# with [`modal.concurrent`](https://modal.com/docs/reference/modal.concurrent).
# For details, see [the guide](https://modal.com/docs/guide/concurrent-inputs).

TARGET_INPUTS = 10

# Generally, this choice needs to be made as part of
# [LLM inference engine benchmarking](https://modal.com/llm-almanac/how-to-benchmark).

# ### Controlling container lifecycles with `modal.Cls`

# We wrap up all of the choices we made about the infrastructure
# of our inference server into a number of Python decorators
# that we apply to a Python class that encapsulates the logic
# to run our server.

# The key decorators are:

# - [`@app.cls`](https://modal.com/docs/guide/lifecycle-functions) to define the core of our service.
# We attach our Image, request a GPU, attach our cache Volumes, specify the region, and configure auto-scaling.
# See [the reference documentation](https://modal.com/docs/reference/modal.App#cls) for details.

# - `@modal.experimental.http_server` to turn our Python code into an HTTP server
# (i.e. fronting all of our containers with a proxy with a URL). The wrapped code
# needs to eventually listen for HTTP connections on the provided `port`.

# - [`@modal.concurrent`](https://modal.com/docs/guide/concurrent-inputs) to specify how many
# requests our server can handle before we need to scale up.

# - [`@modal.enter` and `@modal.exit`](https://modal.com/docs/guide/lifecycle-functions) to indicate
# which methods of the class should be run when starting the server and shutting it down.

# Modal considers a new replica ready to receive inputs once the `modal.enter` methods have exited
# and the container accepts connections.
# To ensure that we actually finish setting up our server before we are marked ready for inputs,
# we define a helper function to check whether the server is finished setting up and to
# send it a few test inputs.

# We use the [`requests` library](https://requests.readthedocs.io/en/latest/)
# to send ourselves these HTTP requests on
# [`localhost`/`127.0.0.1`](https://superuser.com/questions/31824/why-is-localhost-ip-127-0-0-1).

with sglang_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = 5 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(5)
    raise TimeoutError(f"SGLang server not ready within {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def warmup():
    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=10
        ).raise_for_status()


# With all this in place, we are ready to define our high-performance, low-latency
# LLM inference server.

app = modal.App(name="example-sglang-low-latency")
PORT = 8000


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
    region=REGION,
    min_containers=MIN_CONTAINERS,
)
@modal.experimental.http_server(
    port=PORT,  # wrapped code must listen on this port
    proxy_regions=[REGION],  # location of proxies, should be same as Cls region
    exit_grace_period=5,  # seconds, time to finish up requests when closing down
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class SGLang:
    @modal.enter()
    def startup(self):
        """Start the SGLang server and block until it is healthy, then warm it up and put it to sleep."""
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            f"{PORT}",
            "--tp",  # use all GPUs to split up tensor-parallel operations
            f"{N_GPUS}",
            "--cuda-graph-max-bs",  # only capture CUDA graphs for batch sizes we're likely to observe
            f"{TARGET_INPUTS * 2}",
            "--enable-metrics",  # expose metrics endpoints for telemetry
            "--decode-log-interval",  # how often to log during decoding, in tokens
            "100",
            "--mem-fraction",  # leave space for speculative model
            "0.8",
        ]

        cmd += [  # add speculative config
            item for k, v in speculative_config.items() for item in (f"--{k}", str(v))
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
# modal deploy sglang_low_latency.py
# ```

# This will create a new App on Modal and build the container image for it if it hasn't been built yet.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-sglang-low-latency-sglang.us-east.modal.direct`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-sglang-low-latency-sglang.us-east.modal.direct/docs`.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands.
# For simple routes, you can even send a request directly from the docs page.

# Note: when no replicas are available, Modal will respond with
# the [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503).
# In your browser, you can just hit refresh until the docs page appears.
# You can see the status of the applicaton and its containers on your [Modal dashboard](https://modal.com/apps).

# ## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that hits the server with a simple client.

# If you execute the command

# ```bash
# modal run sglang_low_latency.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, prompt=None, twice=True):
    url = SGLang._experimental_get_flash_urls()[0]

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if prompt is None:
        prompt = "Explain the Singular Value Decomposition."

    content = [{"type": "text", "text": prompt}]

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    await probe(url, messages, timeout=test_timeout)
    if twice:
        messages[0]["content"] = "You are Jar Jar Binks."
        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await probe(url, messages, timeout=1 * MINUTES)


# This test relies on the two helper functions below,
# which ping the server and wait for a valid response to stream.

# The `probe` helper function specifically ignores
# two types of errors that can occur while a replica
# is starting up -- timeouts on the client and 5XX responses from the server.
# Modal returns the [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)
# when an `experimental.http_server` has no live replicas.


async def probe(url, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [{"role": "user", "content": "Tell me a joke."}]

    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url) as session:
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
    payload = {"messages": messages, "stream": True}
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
        print(full_text)
