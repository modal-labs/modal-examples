# ---
# deploy: true
# cmd: ["python", "06_gpu_and_ml/llm-serving/vllm_low_latency.py"]
# ---

# # Low latency Qwen 3.5 35B-A3B with DFlash speculative decoding and vLLM on Modal

# In this example, we show how to serve [vLLM](https://docs.vllm.ai) at low latency on Modal.

# This example is intended to demonstrate everything required to run
# inference at the highest performance and with the lowest latency possible,
# and so it includes advanced features of both vLLM and Modal.
# For a simpler introduction to LLM serving, see
# [this example](https://modal.com/docs/examples/llm_inference).

# To minimize routing overheads, we use `@modal.experimental.http_server`,
# which uses a new, low-latency routing service on Modal designed for latency-sensitive inference workloads.
# This gives us more control over routing, but with increased power comes increased responsibility.

# We also include instructions for cutting cold start times by an order of magnitude using Modal's
# [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot).

# ## Speculative decoding with DFlash

# [DFlash](https://arxiv.org/abs/2602.06036) is a speculative decoding method that uses
# a lightweight block diffusion model to draft multiple tokens in parallel,
# achieving up to 2.8x speedup over autoregressive decoding.
# It works by generating draft tokens with a small diffusion model,
# then verifying them against the main model in a single forward pass.
# The more tokens that are accepted, the faster decoding proceeds.

# For low latency serving, speculative decoding is critical:
# it reduces the time-to-first-token and increases tokens-per-second
# without sacrificing output quality, since every drafted token is verified.

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/images).
# We'll use the [vLLM inference server](https://docs.vllm.ai).
# vLLM can be installed with `uv pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

# We install vLLM from the nightly wheel index, which is required for DFlash support.
# The `--torch-backend=auto` flag ensures the correct PyTorch backend for our CUDA version.

# While we're at it, we import the dependencies we'll need both remotely and locally (for deployment).

import asyncio
import json
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60  # seconds

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install(
        "vllm",
        "huggingface-hub",
        extra_index_url="https://wheels.vllm.ai/nightly",
    )
    .env(
        {
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCH_CPP_LOG_LEVEL": "FATAL",
        }
    )
)

# We also choose a [GPU](https://modal.com/docs/guide/gpu) to deploy our inference server onto.
# We choose the [H100 GPU](https://modal.com/blog/introducing-h100),
# which offers excellent price-performance.
# The H100's 80 GB of GPU memory fits the Qwen3.5-35B-A3B model (~66 GB in BF16),
# the DFlash draft model (~1 GB), and a KV cache sufficient for 32K context.
# We restrict the maximum model length with `--max-model-len 32768` to keep
# KV cache memory within the H100's capacity while still supporting
# most single-turn and short multi-turn interactions.

GPU = "H100"

# ### Loading and caching the model weights

# We serve the [Qwen 3.5 35B-A3B model](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) in BF16.
# Despite having 35B total parameters, this is a
# [Mixture-of-Experts (MoE)](https://modal.com/gpu-glossary/model-architecture/mixture-of-experts)
# model that activates only 3B parameters per token,
# giving it the inference cost of a much smaller model with the quality of a much larger one.

# We use BF16 rather than the FP8-quantized variant because the DFlash draft model
# shares the target model's embedding and LM head weights. The quantization mismatch
# between an FP8 target model and a BF16 draft model corrupts draft token generation,
# causing DFlash acceptance rates to collapse from ~40% to 0% and producing garbled output.
# The officially tested combination for DFlash is the BF16 model.

# For speculative decoding, we pair it with the
# [DFlash draft model](https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash),
# a lightweight block diffusion model that generates draft tokens in parallel.

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = (  # pin revision id to avoid nasty surprises!
    "ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"  # latest commit as of 2026-04-07
)

DRAFT_MODEL_NAME = "z-lab/Qwen3.5-35B-A3B-DFlash"

# We load the model [from the Hugging Face Hub](https://huggingface.co/collections/Qwen/qwen3),
# so we'll need their Python package.

# We don't want to load the model from the Hub every time we start the server.
# We can load it much faster from a [Modal Volume](https://modal.com/docs/guide/volumes).
# Typical speeds are around one to two GB/s.

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
MODEL_PATH = f"{HF_CACHE_PATH}/{MODEL_NAME}"

# vLLM's `torch.compile` step takes over 10 minutes on first run.
# We persist the compile cache to a separate Volume so that subsequent
# cold starts skip compilation entirely, reducing startup from 15+ minutes
# to under 2 minutes (just model weight loading).
VLLM_CACHE_VOL = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)
VLLM_CACHE_PATH = "/root/.cache/vllm"

# In addition to pointing the Hugging Face Hub at the path
# where we mount the Volume, we also
# [turn on "high performance" downloads](https://huggingface.co/docs/hub/en/models-downloading#faster-downloads),
# which can fully saturate our network bandwidth.

vllm_image = vllm_image.env(
    {"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"}
)

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

# Latencies for multi-turn interactions with LLMs are
# substantially cut when previous interaction turns are in the KV cache.
# KV caches are stored in [GPU RAM](https://modal.com/gpu-glossary/device-hardware/gpu-ram),
# so they aren't shared across replicas.
# To improve cache hit rate, `modal.experimental.http_server`
# includes sticky routing based on a client-provided header.
# See the client code below for details.

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

TARGET_INPUTS = 20

# Generally, this choice needs to be made as part of
# [LLM inference engine benchmarking](https://modal.com/llm-almanac/how-to-benchmark).

# ### Cutting cold starts with GPU memory snapshots

# vLLM supports a sleep mode that allows us to leverage Modal's
# [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot)
# for dramatically faster cold starts.

# When `enable_memory_snapshot=True` and `experimental_options={"enable_gpu_snapshot": True}`
# are set on the class, Modal captures both CPU and GPU memory state.
# The `@modal.enter(snap=True)` method runs before the snapshot is taken:
# we start vLLM, wait for it to be ready, warm it up, then put it to sleep.
# The `@modal.enter(snap=False)` method runs after restoring from snapshot:
# we wake vLLM back up so it can serve requests immediately.

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
# To ensure that we actually finish setting up our server before we are marked ready for inputs,
# we define a helper function to check whether the server is finished setting up and to
# send it a few test inputs.

# We use the [`requests` library](https://requests.readthedocs.io/en/latest/)
# to send ourselves these HTTP requests on
# [`localhost`/`127.0.0.1`](https://superuser.com/questions/31824/why-is-localhost-ip-127-0-0-1).

with vllm_image.imports():
    import requests

PORT = 8000


def wait_ready(process: subprocess.Popen, timeout: int = 30 * MINUTES):
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
    raise TimeoutError(f"vLLM server not ready within {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def warmup():
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=payload,
            timeout=5 * MINUTES,
        ).raise_for_status()


def sleep(level: int = 1):
    requests.post(f"http://127.0.0.1:{PORT}/sleep?level={level}").raise_for_status()


def wake_up():
    requests.post(f"http://127.0.0.1:{PORT}/wake_up").raise_for_status()


# With all this in place, we are ready to define our high-performance, low-latency
# LLM inference server.

APP_NAME = "example-vllm-low-latency"
app = modal.App(name=APP_NAME)

SPECULATIVE_CONFIG = json.dumps(
    {
        "method": "dflash",
        "model": DRAFT_MODEL_NAME,
        "num_speculative_tokens": 15,
    }
)


@app.cls(
    image=vllm_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, VLLM_CACHE_PATH: VLLM_CACHE_VOL},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    region=REGION,
    min_containers=MIN_CONTAINERS,
    timeout=30 * MINUTES,
)
@modal.experimental.http_server(
    port=PORT,  # wrapped code must listen on this port
    proxy_regions=[REGION],  # location of proxies, should be same as Cls region
    exit_grace_period=5,  # seconds, time to finish up requests when closing down
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class VLLM:
    @modal.enter(snap=True)
    def startup(self):
        """Start the vLLM server and block until it is healthy, then warm it up and put it to sleep."""

        # DFlash speculative decoding: the draft model generates tokens in
        # parallel via block diffusion, then the main model verifies them in a
        # single forward pass — achieving up to 2.8x speedup.
        # Flash attention backend is required for DFlash compatibility and is
        # generally the fastest attention implementation available.
        # --max-num-batched-tokens limits scheduling overhead to improve
        # per-request latency at the cost of peak throughput.
        # --reasoning-parser qwen3 separates thinking content from the final
        # response in the API output for Qwen3.5's default thinking mode.
        # --language-model-only skips the vision encoder to free GPU memory for
        # KV cache, since we only serve text-only requests.
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            f"{PORT}",
            "--uvicorn-log-level",
            "error",
            "--disable-uvicorn-access-log",
            "--enable-sleep-mode",
            "--speculative-config",
            SPECULATIVE_CONFIG,
            "--attention-backend",
            "flash_attn",
            "--max-num-batched-tokens",
            "32768",
            "--max-model-len",
            "32768",
            "--reasoning-parser",
            "qwen3",
            "--language-model-only",
        ]

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()
        sleep(1)

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
# modal deploy vllm_low_latency.py
# ```

# This will create a new App on Modal and build the container image for it if it hasn't been built yet.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-vllm-low-latency-vllm.us-east.modal.direct`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-vllm-low-latency-vllm.us-east.modal.direct/docs`.
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
# modal run vllm_low_latency.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=30 * MINUTES, prompt=None, twice=True):
    url = VLLM._experimental_get_flash_urls()[0]

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

    client_id = str(0)  # set this to some string per multi-turn interaction
    # often a UUID per "conversation"
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
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        full_text = ""
        reasoning_text = ""

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
            reasoning_chunk = delta.get("reasoning_content")

            if reasoning_chunk:
                print(reasoning_chunk, end="", flush=True)
                reasoning_text += reasoning_chunk

            if chunk:
                if reasoning_text:
                    print("\n---\n", end="", flush=True)
                    reasoning_text = ""
                print(chunk, end="", flush=True)
                full_text += chunk
        print()  # newline after stream completes
        print(full_text)


if __name__ == "__main__":
    # after deployment, we can use the class from anywhere
    vllm_server = modal.Cls.from_name(APP_NAME, "VLLM")

    async def main(url):
        messages = [{"role": "user", "content": "Tell me a joke."}]
        await probe(url, messages, timeout=10 * MINUTES)

    print("calling inference server")
    asyncio.run(main(vllm_server._experimental_get_flash_urls()[0]))
