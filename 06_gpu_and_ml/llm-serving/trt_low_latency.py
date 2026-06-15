# # Low latency LLaMA 3 8B with TensorRT-LLM and Modal

# In this example, we show how to serve [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
# at low latency on Modal.

# TensorRT-LLM achieves exceptional inference latency through careful engine optimization.
# Combined with Modal's `@modal.experimental.http_server` for low-latency routing,
# we can serve models with minimal overhead between client and GPU.

# For a deeper dive into TensorRT-LLM engine tuning, including speculative decoding,
# see [the latency-optimized TRT-LLM example](https://modal.com/docs/examples/trtllm_latency).

# ## Set up the container image

# We start from an official `nvidia/cuda` image and install TensorRT-LLM on top.
# For details on the installation process, see
# [the TRT-LLM latency example](https://modal.com/docs/examples/trtllm_latency).

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
    add_python="3.12",
).entrypoint([])  # silence noisy NVIDIA license logging

tensorrt_image = tensorrt_image.apt_install(
    "openmpi-bin", "libopenmpi-dev", "git", "git-lfs", "wget"
).pip_install(
    "tensorrt-llm==0.18.0",
    "pynvml<12",
    "flashinfer-python==0.2.5",
    "cuda-python==12.9.1",
    "onnx==1.19.1",
    pre=True,
    extra_index_url="https://pypi.nvidia.com",
)

# ## Cache model weights in a Modal Volume

# We serve [Meta's LLaMA 3 8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct),
# loading it from a [Modal Volume](https://modal.com/docs/guide/volumes)
# for fast startup on subsequent runs.

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
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

# We run on a single [H100 GPU](https://modal.com/docs/guide/gpu)
# with FP8 quantization and low-latency GEMM plugins.
# FP8 halves the data moved per operation compared to FP16,
# and the low-latency plugins from NVIDIA's
# [CUTLASS library](https://github.com/NVIDIA/cutlass)
# provide optimized kernels for small batch sizes.

# For a full walkthrough of these engine tuning options,
# see [the TRT-LLM latency example](https://modal.com/docs/examples/trtllm_latency).

N_GPUS = 1
GPU = f"H100:{N_GPUS}"
MAX_BATCH_SIZE = 1  # minimize latency by processing one request at a time

# ## Define the inference server

# To minimize routing latency, we use
# [`@modal.experimental.http_server`](https://modal.com/docs/guide/http-server),
# which routes requests through a low-latency proxy close to our GPU containers.

# We select the `us` [cloud region](https://modal.com/docs/guide/region-selection)
# and a nearby proxy region.

REGION = "us"
PORT = 8000
PROXY_REGION = "us-west"
MIN_CONTAINERS = 0  # set to 1 to ensure one replica is always ready
TARGET_INPUTS = 1  # low concurrency for minimum per-request latency

# ### Health check and warmup helpers

# We poll the server's `/health` endpoint until it's ready,
# then send a few warm-up requests so the first real request isn't slow.

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
            subprocess.CalledProcessError,
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

# We use [`@app.cls`](https://modal.com/docs/guide/lifecycle-functions) to manage
# the server lifecycle. On the first container start, we build an optimized engine
# using the TensorRT-LLM [Python API](https://nvidia.github.io/TensorRT-LLM/llm-api)
# with FP8 quantization and low-latency plugins, then cache it in the Volume.
# Subsequent starts load the cached engine in seconds and launch `trtllm-serve`
# to expose an OpenAI-compatible HTTP API.

app = modal.App("example-trt-low-latency")


@app.cls(
    image=tensorrt_image,
    gpu=GPU,
    volumes={VOLUME_PATH: volume},
    region=REGION,
    min_containers=MIN_CONTAINERS,
    startup_timeout=20 * MINUTES,
)
@modal.experimental.http_server(
    port=PORT,
    proxy_regions=[PROXY_REGION],
    exit_grace_period=15,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class TRT:
    @modal.enter()
    def startup(self):
        """Download model, build/load the optimized engine, and start trtllm-serve."""
        from huggingface_hub import snapshot_download
        from tensorrt_llm import LLM, BuildConfig
        from tensorrt_llm.llmapi import CalibConfig, QuantConfig
        from tensorrt_llm.plugin.plugin import PluginConfig

        model_path = str(MODELS_PATH / MODEL_ID)
        engine_path = str(MODELS_PATH / MODEL_ID / "trtllm_engine" / "serve")

        snapshot_download(
            MODEL_ID,
            local_dir=model_path,
            ignore_patterns=["*.pt", "*.bin"],
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
        else:
            print(f"loading cached engine from {engine_path}")

        cmd = [
            "trtllm-serve",
            engine_path,
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

# To deploy the server on Modal, run

# ```bash
# modal deploy trt_low_latency.py
# ```

# Once deployed, you'll see a URL like
# `https://your-workspace--example-trt-low-latency-trt.us-west.modal.direct`.

# Interactive [Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# are available at the `/docs` route of that URL.

# Note: when no replicas are running, Modal returns a
# [503 Service Unavailable](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503).
# Refresh until the docs page appears, or check the
# [Modal dashboard](https://modal.com/apps) for container status.

# ## Test the server

# To test the server, run

# ```bash
# modal run trt_low_latency.py
# ```

# This spins up a fresh replica on Modal and runs the client code below locally.


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, prompt=None, twice=True):
    url = (await TRT._experimental_get_flash_urls.aio())[0]

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


# The `probe` helper retries on 503s (no live replicas) and timeouts.

# We include a `Modal-Session-ID` header for
# [sticky routing](https://modal.com/docs/guide/concurrent-inputs),
# which improves KV cache hit rates for multi-turn conversations.


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
