# # Low latency Nvidia Nemotron 3 Nano 30B with SGLang and Modal

# In this example, we show how to serve Nvidia's
# [Nemotron 3 Nano 30B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
# on Modal at low latency with [SGLang](https://github.com/sgl-project/sglang).

# The Nemotron 3 Nano 30B model has 30 billion total parameters with only 3 billion active per token,
# thanks to its sparse Mixture-of-Experts (MoE) architecture combined with
# hybrid Mamba-2/Transformer attention layers.
# You can read more in the paper [here](https://arxiv.org/abs/2512.20856).

# To minimize routing overheads, we use `@modal.experimental.http_server`,
# which uses a new, low-latency routing service on Modal designed for latency-sensitive inference workloads.

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/images).

# We start from a container image provided
# [by the SGLang team via Dockerhub](https://hub.docker.com/r/lmsysorg/sglang/tags).

import asyncio
import json
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60  # seconds

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.11")
    .entrypoint(  # silence chatty logs on container start
        []
    )
    .run_commands(  # clean up Image
        "rm -rf /root/.cache/huggingface"
    )
)

# ### Loading and caching the model weights

# We'll serve [NVIDIA's Nemotron 3 Nano 30B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16).
# This model has 30 billion total parameters but only 3 billion are active per token,
# using a sparse MoE architecture with 128 routed experts (6 active per token)
# and a hybrid Mamba-2/Transformer design for efficient inference.
# At BF16 precision, the model weights are around 63 GB,
# fitting on a single H100 GPU.

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
MODEL_REVISION = (  # pin revision to avoid surprises if upstream updates
    "cbd3fa9f933d55ef16a84236559f4ee2a0526848"  # latest commit as of 2026-06-22
)

# We load the model [from the Hugging Face Hub](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16).
# Downloads from the Hub are much faster if you are authenticated.
# So we add a Hugging Face token as a [Modal Secret](https://modal.com/docs/guide/secrets).
# You can create a Modal Secret with your Hugging Face token
# [here](https://modal.com/secrets). Make sure to name it `huggingface-secret`!

hf_secret = modal.Secret.from_name("huggingface-secret")

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

# We also choose a [GPU](https://modal.com/docs/guide/gpu) to deploy our inference server onto.
# Since the 30B model weights are ~63 GB at BF16, they fit on a single H100 (80 GB).
# With only 3B active parameters per forward pass, latency remains excellent
# even on a single GPU.

GPU_TYPE, N_GPUS = "H100", 1
GPU = f"{GPU_TYPE}:{N_GPUS}"

# ## Define the inference server and infrastructure

# ### Selecting infrastructure to minimize latency

# Minimizing latency requires geographic co-location of clients and servers.

# So for low latency LLM inference services on Modal, you must select a
# [cloud region](https://modal.com/docs/guide/region-selection)
# for both the GPU-accelerated containers running inference
# and for the internal Modal proxies that forward requests to them
# as part of defining a `modal.experimental.http_server`.

REGION = "us"
PROXY_REGION = "us-west"

# For production-scale LLM inference services, there are generally
# enough requests to justify keeping at least one replica running at all times.
# Having a "warm" or "live" replica reduces latency by skipping slow initialization work
# that occurs when new replica boots up (a ["cold start"](https://modal.com/docs/guide/cold-start)).

# To ensure at least one container is always available,
# we can set the `min_containers` of our Modal Function
# to `1` or more.

# However, since this is documentation code, we'll set it to `0`
# to avoid surprise bills during casual use.

MIN_CONTAINERS = 0  # set to 1 to ensure one replica is always ready

# We set a target for the number of inputs to run on a single container
# with [`modal.concurrent`](https://modal.com/docs/reference/modal.concurrent).
# For details, see [the guide](https://modal.com/docs/guide/concurrent-inputs).

TARGET_INPUTS = 16

# ### Controlling container lifecycles with `modal.Cls`

# We wrap up all of the choices we made about the infrastructure
# of our inference server into a number of Python decorators
# that we apply to a Python class that encapsulates the logic
# to run our server.

# Modal considers a new replica ready to receive inputs once the `modal.enter` methods have exited
# and the container accepts connections.
# To ensure that we actually finish setting up our server before we are marked ready for inputs,
# we define a helper function to check whether the server is finished setting up and to
# send it a few test inputs.

with sglang_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = 20 * MINUTES):
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


# ### Server configuration

# We configure the SGLang server with settings appropriate for
# the Nemotron 3 Nano 30B model. Since BF16 weights are ~63 GB on an 80 GB H100,
# we reduce the static memory fraction to leave room for the CUDA runtime
# while still allocating enough for KV cache.

server_args = [
    "--context-length",
    "131072",
    "--mem-fraction-static",
    "0.85",
    "--chunked-prefill-size",
    "32768",
    "--trust-remote-code",
]

# With all this in place, we are ready to define our low-latency
# Nemotron 3 Nano 30B inference server.

app = modal.App(name="example-nemotron-30b-inference")
PORT = 8000


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    region=REGION,
    min_containers=MIN_CONTAINERS,
    secrets=[hf_secret],
    startup_timeout=20 * MINUTES,  # time to load weights
)
@modal.experimental.http_server(
    port=PORT,  # wrapped code must listen on this port
    proxy_regions=[PROXY_REGION],  # location of proxies, should overlap with Cls region
    exit_grace_period=15,  # seconds, time to finish up requests when closing down
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class Server:
    @modal.enter()
    def startup(self):
        """Start the SGLang server and block until it is healthy, then warm it up."""

        cmd = [
            "sglang",
            "serve",
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
            "--cuda-graph-max-bs",
            f"{TARGET_INPUTS * 2}",
            "--enable-metrics",
            "--decode-log-interval",
            "10",
        ] + server_args

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Deploy the server

# To deploy the server on Modal, just run

# ```bash
# modal deploy nemotron_30b_inference.py
# ```

# This will create a new App on Modal and build the container image for it if it hasn't been built yet.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-nemotron-30b-inference-server.modal.direct`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL.

# ## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that hits the server with a simple client.

# If you execute the command

# ```bash
# modal run nemotron_30b_inference.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.


@app.local_entrypoint()
async def test(test_timeout=120 * MINUTES, prompt=None, twice=True):
    url = (await Server._experimental_get_flash_urls.aio())[0]

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
        await probe(url, messages, timeout=10 * MINUTES)


# This test relies on the two helper functions below,
# which ping the server and wait for a valid response to stream.

# We include a header with each request --
# `Modal-Session-ID`.
# This header is used by clients of `http_server`s on Modal
# to identify which requests should be routed to the same container.
# Set this to a different value per distinct multi-turn interaction
# to improve KV cache hit rates.


async def probe(url, messages=None, timeout=20 * MINUTES):
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
            chunk = delta.get("content") or delta.get("reasoning_content")

            if chunk:
                print(chunk, end="", flush="\n" in chunk or "." in chunk)
                full_text += chunk
        print()  # newline after stream completes
