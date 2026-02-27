# # Serve the Qwen3.5 Vision-Language Model with SGLang

# Vision-Language Models (VLMs) are like LLMs with eyes:
# they can generate text based not just on other text,
# but on images as well.

# This example shows how to serve a VLM on Modal using the
# [SGLang](https://github.com/sgl-project/sglang) library
# with an OpenAI-compatible API server.

# ## Setup and container image definition

# First, we import our global dependencies
# and define constants.

import asyncio
import json
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60

# To define the container [Image](https://modal.com/docs/guide/images)
# with our server's dependencies,
# we build off of the official SGLang Docker image with CUDA 12.9.

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.9-cu129-amd64-runtime")
    .entrypoint([])
    .uv_pip_install("huggingface-hub==0.36.0")
)

# ## Configure the model

# [Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)
# is a vision-language reasoning foundational model with 35B total parameters,
# of which only 3B are activated per input sequence per forward pass.
# We use the [8bit quantized floating point](https://quant.exposed)
# version of the model for faster [cold starts](https://modal.com/docs/guide/cold-start)
# and faster inference with negligible behavior differences.

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"
MODEL_REVISION = "0b2752837483aa34b3db6e83e151b150c0e00e49"

# ## Configure GPU

# We use a single H100 GPU. The 35 GB of model weights fits comfortably in this GPU's 80GB of
# [high-bandwidth memory](https://modal.com/gpu-glossary/device-hardware/gpu-ram).

GPU = "H100!:1"
N_GPUS = 1

# ## Cacheing in Modal Volumes

# Modal Apps typically cache some artifacts in a [Modal Volume](https://modal.com/docs/guide/volumes)
# for faster cold starts.
# Here, we cache the model weights and the JIT-compiled DeepGEMM kernels.

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

DG_CACHE_VOL = modal.Volume.from_name("deepgemm-cache", create_if_missing=True)
DG_CACHE_PATH = "/root/.cache/deepgemm"

sglang_image = sglang_image.env(
    {
        "HF_HUB_CACHE": HF_CACHE_PATH,
        "HF_XET_HIGH_PERFORMANCE": "1",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "1",
    }
)

# We additionally compile the DeepGEMM kernels as part of building the container
# [Image](https://modal.com/docs/guide/images).
# This can take tens of minutes the first time, but only takes seconds when reading from cache.


def compile_deep_gemm():
    import os
    import subprocess

    if int(os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1")):
        subprocess.run(
            f"python3 -m sglang.compile_deep_gemm --model-path {MODEL_NAME} --revision {MODEL_REVISION} --tp {N_GPUS}",
            shell=True,
            check=True,
        )


sglang_image = sglang_image.run_function(
    compile_deep_gemm,
    volumes={DG_CACHE_PATH: DG_CACHE_VOL, HF_CACHE_PATH: HF_CACHE_VOL},
    gpu=GPU,
)

# ## Define the inference server

# With environment setup out of the way, we're ready to define our inference server.
# We use a [Modal Cls](https://modal.com/docs/guide/lifecycle-functions)
# to separate container startup logic from input processing
# (as part of `modal.enter`-decorated methods).
# We use a Modal HTTP Server to create a low latency edge deployment
# in the `us` served by a proxy in `us-east`.
# We also handle clean teardown of the server in a `modal.exit` method.


REGION = "us"
PROXY_REGION = "us-east"

PORT = 8000
TARGET_INPUTS = 10

app = modal.App(name="example-sglang-vlm")


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
    region=REGION,
    timeout=15 * MINUTES,
)
@modal.experimental.http_server(port=PORT, proxy_regions=[PROXY_REGION])
@modal.concurrent(target_inputs=TARGET_INPUTS)
class VlmServer:
    @modal.enter()
    def startup(self):
        self.process = _start_server()
        wait_ready(self.process)
        warmup()

    @modal.exit()
    def stop(self):
        self.process.terminate()
        self.process.wait()


# ### Setting up the server

# The server configuration is based on the information in the
# [Hugging Face repo](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8).
# It includes speculative decoding via multi-token prediction
# for improved performance at low to moderate concurrency.
# For more on optimizing the performance of VLMs and LLMs,
# see [this guide](https://modal.com/docs/guide/high-performance-llm-inference).


def _start_server() -> subprocess.Popen:
    """Start SGLang server in a subprocess"""
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
        "--tp",
        f"{N_GPUS}",
        "--cuda-graph-max-bs",
        f"{TARGET_INPUTS * 2}",
        "--enable-metrics",
        "--mem-fraction-static",
        "0.8",
        "--context-length",
        "131_072",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--speculative-algo",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]

    print("Starting SGLang server with command:")
    print(*cmd)

    return subprocess.Popen(" ".join(cmd), shell=True, start_new_session=True)


# Before returning from our `modal.enter` method,
# we wait for the server to finish spinning up, which can take several minutes.


def wait_ready(process: subprocess.Popen, timeout: int = 10 * MINUTES):
    import requests

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


# We also send a few warmup requests to ensure
# that the server is fully ready to service requests --
# otherwise the first few requests to a new replica might be
# substantially slower.

SAMPLE_PAYLOAD = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://modal-cdn.com/golden-gate-bridge.jpg"
                    },
                },
                {"type": "text", "text": "What is this?"},
            ],
        }
    ],
    "max_tokens": 16,
}


def warmup():
    import requests

    for _ in range(2):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            json=SAMPLE_PAYLOAD,
            timeout=120,
        ).raise_for_status()


# ## Test the server

# We can test the entire server creation, from soup to nuts,
# by running the file with `modal run`.
# We just need to add a `local_entrypoint` that exercises the server.


@app.local_entrypoint()
async def main():
    url = (await VlmServer._experimental_get_flash_urls.aio())[0]

    messages = SAMPLE_PAYLOAD["messages"]
    print(f"Sending image at {messages[0]['content'][0]['image_url']} to the server")

    await probe(url, messages, timeout=10 * MINUTES)


# The client logic is normally handled by your preferred interface --
# a coding agent harness like [OpenCode](https://modal.com/docs/examples/opencode_server),
# a chat UI in the browser. Our server uses the standard OpenAI-compatible API format,
# so most of these clients should work out of the box.
# We replicate the minimum amount of its functionality we need for a test below.

# Note that in the `probe` we include a `Modal-Session-Id` header for sticky routing
# between Modal HTTP Server replicas and ignore 503s that occur
# when no Modal HTTP Server replicas are available.


async def probe(url: str, messages: list, timeout: int = 5 * MINUTES):
    headers = {"Modal-Session-Id": "test-session"}
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
                raise
    raise TimeoutError(f"No response from server within {timeout} seconds")


async def _send_request_streaming(
    session: aiohttp.ClientSession, messages: list, timeout: int | None = None
) -> None:
    payload = {
        "messages": messages,
        "stream": True,
        "top_k": 20,
    }
    headers = {"Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        full_text = ""

        chunk = ""
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
            chunk += delta.get("content") or delta.get("reasoning_content") or ""

            if chunk and ("." in chunk or "\n" in chunk):
                print(chunk, end="", flush=True)
        if chunk:
            print(chunk, end="", flush=True)
            full_text += chunk
        print()
        return full_text

        print()
        return full_text


# You can kick off a test run with the command

# ```bash
# modal run sglang_vlm.py
# ```

# ## Deploy the server

# When you're ready to deploy the server,
# replace `modal run` with `modal deploy`:

# ```bash
# modal deploy sglang_vlm.py
# ```
