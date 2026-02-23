# ---
# deploy: true
# cmd: ["python", "06_gpu_and_ml/llm-serving/lfm_snapshot.py"]
# ---

# # Low Latency, Serverless LFM 2 with vLLM and Modal

# In this example, we show how to serve Liquid AI's [LFM 2 models](https://www.liquid.ai/liquid-foundation-models)
# with [vLLM](https://docs.vllm.ai) with low latency and fast cold starts on Modal.

# The LFM 2 models are not vanilla Transformers -- they have a hybrid architecture,
# discovered via an architecture search that optimized for quality, latency, and memory footprint.
# Check out their [technical report](https://arxiv.org/abs/2511.23404v1)
# for more details.

# This example demonstrates techniques to run inference at high efficiency,
# including advanced features of both vLLM and Modal.
# For a simpler introduction to LLM serving, see
# [this example](https://modal.com/docs/examples/llm_inference).

# To minimize routing overheads, we use `@modal.experimental.http_server`,
# which uses a new, low-latency routing service on Modal designed for latency-sensitive inference workloads.
# This gives us more control over routing, but with increased power comes increased responsibility.

# We also include instructions for cutting cold start times by an order of magnitude using Modal's
# [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot).

# Fast cold starts are particularly useful for LLM inference applications
# that have highly "bursty" workloads, like document processing.
# See [this guide](https://modal.com/docs/guide/high-performance-llm-inference)
# for a breakdown of different LLM inference workloads and how to optimize them.

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/images).
# We'll use the [vLLM inference server](https://docs.vllm.ai).

# While we're at it, we import the dependencies we'll need both remotely and locally (for deployment).

import asyncio
import json
import os
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60

MODEL_NAME = os.environ.get("MODEL_NAME", "LiquidAI/LFM2-8B-A1B")
print(f"Running deployment script for model: {MODEL_NAME}")

vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.15.1")
    .entrypoint([])
    .run_commands("ln -s $(which python3) /usr/bin/python")
    .pip_install("transformers==5.1.0")
    .env(
        {
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCH_CPP_LOG_LEVEL": "FATAL",
            "MODEL_NAME": MODEL_NAME,
        }
    )
)

# ### Selecting the GPU

# We choose the [H100 GPU](https://modal.com/blog/introducing-h100),
# which offers excellent price-performance and has sufficient VRAM to store the models.

N_GPU = 1
GPU = "H100"

# ### Loading and caching the model weights

# We don't want to load the model from the Hub every time we start the server.
# We can load it much faster from a [Modal Volume](https://modal.com/docs/guide/volumes).
# Typical speeds are around one to two GB/s.

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# In addition to pointing the Hugging Face Hub at the path
# where we mount the Volume, we also
# [turn on "high performance" downloads](https://huggingface.co/docs/hub/en/models-downloading#faster-downloads),
# which can fully saturate our network bandwidth,
# and provide an `HF_TOKEN` via a [Modal Secret](https://modal.com/docs/guide/secrets)
# so that our downloads aren't throttled.
# You'll need to create a Secret named `huggingface-secret`
# with your token [here](https://modal.com/apps/secrets).

hf_secret = modal.Secret.from_name("huggingface-secret")

# ### Caching compilation artifacts

# Model weights aren't the only thing we want to cache.
# vLLM also produces compilation artifacts that we want to persist across restarts.

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ## Define the inference server and infrastructure

# ### Selecting infrastructure to minimize latency

# Minimizing latency requires geographic co-location of clients and servers.

# So for low latency LLM inference services on Modal, you must select a
# [cloud region](https://modal.com/docs/guide/region-selection)
# for both the GPU-accelerated containers running inference
# and for the internal Modal proxies that forward requests to them
# as part of defining a `modal.experimental.http_server`.

# Here, we assume users are mostly in the northern half of the Americas
# and select the `us-east` cloud region to serve them.
# This should result in at most a few dozen milliseconds of round-trip time.

REGION = "us-east"

# For production-scale LLM inference services, there are generally
# enough requests to justify keeping at least one replica running at all times.
# Having a "warm" or "live" replica reduces latency by skipping slow initialization work
# that occurs when new replica boots up (a ["cold start"](https://modal.com/docs/guide/cold-start)).
# For LLM inference servers, that latency runs from seconds to minutes.

# However, since this is documentation code, we'll set the `min_containers` of our Modal Function
# to `0` to avoid surprise bills during casual use.

MIN_CONTAINERS = 0

# Finally, we need to decide how we will scale up and down replicas
# in response to load. Without autoscaling, users' requests will queue
# when the server becomes overloaded. Even apart from queueing, responses
# generally become slower per user above a certain minimum number of
# concurrent requests.

# So we set a target for the number of inputs to run on a single container
# with [`modal.concurrent`](https://modal.com/docs/reference/modal.concurrent).
# For details, see [the guide](https://modal.com/docs/guide/concurrent-inputs).

# Generally, this choice needs to be made as part of
# [LLM inference engine benchmarking](https://modal.com/llm-almanac/how-to-benchmark).

TARGET_INPUTS = 32
MAX_INPUTS = 100

# ## Speed up cold starts with GPU snapshotting

# Modal is a serverless compute platform, so all of your
# inference services automatically scale up and down to handle
# variable load.

# Scaling up a new replica requires quite a bit of work --
# loading up Python and system packages, loading model weights,
# setting up the inference engine, and so on.

# We can skip over and speed up a bunch of this work
# when spinning up new replicas after the first
# by directly booting from a [memory snapshot](https://modal.com/docs/guide/memory-snapshot),
# which contains the exact in-memory representation of our server just before it begins taking requests.

# Most applications can be snapshot and experience substantial speedups (2x to 10x,
# see [our initial benchmarks here](https://modal.com/blog/gpu-mem-snapshots)).
# However, it generally requires some extra work to adapt the application code.

# vLLM supports a sleep mode that allows us to leverage Modal's
# [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot)
# for dramatically faster cold starts.

# When `enable_memory_snapshot=True` and `experimental_options={"enable_gpu_snapshot": True}`
# are set on the class, Modal captures both CPU and GPU memory state.
# The `@modal.enter(snap=True)` method runs before the snapshot is taken:
# we start vLLM, wait for it to be ready, warm it up, then put it to sleep.
# The `@modal.enter(snap=False)` method runs after restoring from snapshot:
# we wake vLLM back up so it can serve requests immediately.

# ### Sleeping and waking a vLLM server

# We prepare our vLLM inference server for snapshotting by first sending
# a few requests to "warm it up", ensuring that it is fully ready to process requests.
# Then we "put it to sleep", moving non-essential data out of GPU memory,
# with a request to `/sleep`. At this point, we can take a memory snapshot.
# Upon snapshot restoration, we "wake up" the server with a request to `/wake_up`.

# We use the [`requests` library](https://requests.readthedocs.io/en/latest/)
# to send ourselves these HTTP requests on
# [`localhost`/`127.0.0.1`](https://superuser.com/questions/31824/why-is-localhost-ip-127-0-0-1).

VLLM_PORT = 8000

with vllm_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = 15 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{VLLM_PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(5)
    raise TimeoutError(f"vLLM server not ready within {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def warmup():
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=60,
        ).raise_for_status()


def sleep(level: int = 1):
    requests.post(
        f"http://127.0.0.1:{VLLM_PORT}/sleep?level={level}"
    ).raise_for_status()


def wake_up():
    requests.post(f"http://127.0.0.1:{VLLM_PORT}/wake_up").raise_for_status()


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
# The `snap=True`/`snap=False` distinction controls which methods run before/after a memory snapshot.

# Modal considers a new replica ready to receive inputs once the `modal.enter` methods have exited
# and the container accepts connections.

# With all this in place, we are ready to define our high-performance, low-latency
# LFM 2 inference server.

app = modal.App("example-lfm-snapshot")


@app.cls(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=5 * MINUTES,
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[hf_secret],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    region=REGION,
    min_containers=MIN_CONTAINERS,
)
@modal.experimental.http_server(
    port=VLLM_PORT,
    proxy_regions=[REGION],
    exit_grace_period=5,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class LfmVllmInference:
    @modal.enter(snap=True)
    def startup(self):
        """Start the vLLM server and block until it is healthy, then warm it up and put it to sleep."""
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--served-model-name",
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            f"{VLLM_PORT}",
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.8",
            "--max-num-seqs",
            f"{MAX_INPUTS}",
            "--max-cudagraph-capture-size",
            f"{MAX_INPUTS}",
            "--enable-sleep-mode",
        ]

        print(*cmd)
        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()
        sleep(level=1)

    @modal.enter(snap=False)
    def restore(self):
        """Wake vLLM from sleep mode after restoring from a memory snapshot."""
        wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Deploy the server

# To deploy the server on Modal, just run

# ```bash
# modal deploy lfm_snapshot.py
# ```

# This will create a new App on Modal and build the container image for it if it hasn't been built yet.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-lfm-snapshot-lfmvllminference.us-east.modal.direct`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-lfm-snapshot-lfmvllminference.us-east.modal.direct/docs`.
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
# modal run lfm_snapshot.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, prompt=None, twice=True):
    url = (await LfmVllmInference._experimental_get_flash_urls.aio())[0]

    if prompt is None:
        prompt = "List every country and its capital."

    messages = [
        {"role": "user", "content": prompt},
    ]

    await probe(url, messages, timeout=test_timeout)
    if twice:
        messages = [
            {
                "role": "user",
                "content": "List every country and its capital in Chinese.",
            }
        ]
        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await probe(url, messages, timeout=1 * MINUTES)


# This test relies on the `probe` helper function below,
# which ping the server and wait for a valid response to stream.

# The `probe` helper function specifically ignores
# two types of errors that can occur while a replica
# is starting up -- timeouts on the client and 5XX responses from the server.
# Modal returns the [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)
# when an `experimental.http_server` has no live replicas.

# We include a header with each request --
# `Modal-Session-ID`.
# The value associated with this key
# is used to map requests onto containers such that
# while the set of containers is fixed, requests with the same value
# are sent to the same container.
# Set this to a different value per multi-turn interaction
# (prototypically, a user conversation thread with a chatbot)
# to improve KV cache hit rates.
# Note that this header is only compatible with
# Modal `http_server`s.


async def probe(url, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [{"role": "user", "content": "Tell me a joke."}]

    client_id = str(0)  # set this yourself based on KV cache hit-rate
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
    payload = {"model": "llm", "messages": messages, "stream": True}
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

            if not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                continue

            delta = (evt.get("choices") or [{}])[0].get("delta") or {}
            chunk = delta.get("content")

            if chunk:
                print(chunk, end="", flush="\n" in chunk or "." in chunk)
                full_text += chunk
        print()


# ### Test memory snapshotting

# Using `modal run` creates an ephemeral Modal App,
# rather than a deployed Modal App.
# Ephemeral Modal Apps are short-lived,
# so they turn off snapshotting.

# To test the memory snapshot version of the server,
# first deploy it with `modal deploy`
# and then hit it with a client.

# You should observe startup improvements
# after a handful of cold starts
# (usually less than five).
# If you want to see the speedup during a test,
# we recommend heading to the deployed App in your
# [Modal dashboard](https://modal.com/apps)
# and manually stopping containers after they have served a request.

# You can use the client code below to test the endpoint.
# It can be run with the command

# ```
# python lfm_snapshot.py
# ```

if __name__ == "__main__":
    LfmVllmInference = modal.Cls.from_name("example-lfm-snapshot", "LfmVllmInference")

    async def main():
        url = (await LfmVllmInference._experimental_get_flash_urls.aio())[0]
        messages = [{"role": "user", "content": "Tell me ten jokes."}]
        await probe(url, messages, timeout=10 * MINUTES)

    try:
        print("calling inference server")
        asyncio.run(main())
    except modal.exception.NotFoundError as e:
        raise Exception(
            f"To take advantage of GPU snapshots, deploy first with modal deploy {__file__}"
        ) from e
