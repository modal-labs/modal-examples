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
# including advanced features of vLLM and Modal.
# For a simpler introduction to LLM serving, see
# [this example](https://modal.com/docs/examples/llm_inference).

# To minimize routing overheads, we use `@modal.experimental.http_server`,
# which uses a new, low-latency routing service on Modal designed for latency-sensitive inference workloads.

# We also include instructions for cutting cold start times by an order of magnitude using Modal's
# [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/images).
# We'll use the [vLLM inference server](https://docs.vllm.ai).

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

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("examples-lfm-snapshot")

N_GPU = 1
VLLM_PORT = 8000
TARGET_INPUTS = 10
MAX_INPUTS = 100

GPU = "H100"

REGION = "us-east"

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


@app.cls(
    image=vllm_image,
    gpu=GPU,
    scaledown_window=5 * MINUTES,
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret-liquid")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    region=REGION,
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
# modal deploy lfm_snapshot.py
# ```

# This will create a new App on Modal and build the container image for it if it hasn't been built yet.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--examples-lfm-snapshot-lfmvllminference.us-east.modal.direct`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL.

# Note: when no replicas are available, Modal will respond with
# the [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503).
# In your browser, you can just hit refresh until the docs page appears.

# ## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that hits the server with a simple client.

# If you execute the command

# ```bash
# modal run lfm_snapshot.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, prompt=None, twice=True):
    url = LfmVllmInference._experimental_get_flash_urls()[0]

    if prompt is None:
        prompt = "Count to 1000, slowly."

    messages = [
        {"role": "user", "content": prompt},
    ]

    await probe(url, messages, timeout=test_timeout)
    if twice:
        messages = [{"role": "user", "content": "Tell me a joke."}]
        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await probe(url, messages, timeout=1 * MINUTES)


# The `probe` helper function ignores timeouts on the client and 5XX responses from the server
# while a replica is starting up. Modal returns the [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)
# when an `experimental.http_server` has no live replicas.

# We include a header with each request -- `Modal-Session-ID`.
# The value associated with this key is used to map requests onto containers such that
# while the set of containers is fixed, requests with the same value
# are sent to the same container. Set this to a different value per multi-turn interaction
# to improve KV cache hit rates.


async def probe(url, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [{"role": "user", "content": "Tell me a joke."}]

    client_id = str(0)
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
        print(full_text)


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

if __name__ == "__main__":
    LfmVllmInference = modal.Cls.from_name("examples-lfm-snapshot", "LfmVllmInference")

    async def main():
        url = LfmVllmInference._experimental_get_flash_urls()[0]
        messages = [{"role": "user", "content": "Tell me a joke."}]
        await probe(url, messages, timeout=10 * MINUTES)

    try:
        print("calling inference server")
        asyncio.run(main())
    except modal.exception.NotFoundError as e:
        raise Exception(
            f"To take advantage of GPU snapshots, deploy first with modal deploy {__file__}"
        ) from e
