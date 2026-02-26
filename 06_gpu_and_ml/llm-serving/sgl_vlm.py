# # Run Qwen3.5-VL on SGLang for Visual QA

# Vision-Language Models (VLMs) are like LLMs with eyes:
# they can generate text based not just on other text,
# but on images as well.

# This example shows how to run a VLM on Modal using the
# [SGLang](https://github.com/sgl-project/sglang) library
# with an OpenAI-compatible API server.

# ## Setup

import asyncio
import json
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60

# ## Configure the container image

# We use the official SGLang Docker image with CUDA 12.9.

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.9-cu129-amd64-runtime")
    .entrypoint([])
    .uv_pip_install("huggingface-hub==0.36.0")
)

# ## Configure GPU

# We use a single H100 GPU. The FP8-quantized Qwen3.5-35B-A3B model
# fits comfortably in the 80GB H100 memory.

GPU = "H100!:1"
N_GPUS = 1

# ## Configure the model

# [Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8)
# is a vision-language model with 35B total parameters (3B activated)
# in FP8 format for efficient inference.

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"
MODEL_REVISION = "0b2752837483aa34b3db6e83e151b150c0e00e49"

# ## Cache model weights

# We cache model weights in a Modal Volume for faster cold starts.

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

PORT = 8000
TARGET_INPUTS = 10

app = modal.App(name="example-sgl-vlm")


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


def warmup():
    import requests

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
                        },
                    },
                    {"type": "text", "text": "What is this?"},
                ],
            }
        ],
        "max_tokens": 16,
    }
    for _ in range(2):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=120
        ).raise_for_status()


REGION = "us-east"


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
    region=REGION,
    timeout=15 * MINUTES,
)
@modal.experimental.http_server(port=PORT, proxy_regions=[REGION])
@modal.concurrent(target_inputs=TARGET_INPUTS)
class VLM:
    @modal.enter()
    def startup(self):
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
            "65536",
            "--reasoning-parser",
            "qwen3",
        ]

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Test the server


@app.local_entrypoint()
async def main():
    url = (await VLM._experimental_get_flash_urls.aio())[0]

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://modal-public-assets.s3.amazonaws.com/golden-gate-bridge.jpg"
                    },
                },
                {"type": "text", "text": "What is this? Describe it briefly."},
            ],
        }
    ]

    await probe(url, messages, timeout=10 * MINUTES)


async def probe(url: str, messages: list, timeout: int = 5 * MINUTES):
    headers = {"Modal-Session-ID": "test-session"}
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
        "max_tokens": 256,
        "extra_body": {"top_k": 20},
    }
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
                print(chunk, end="", flush=True)
                full_text += chunk

        print()
