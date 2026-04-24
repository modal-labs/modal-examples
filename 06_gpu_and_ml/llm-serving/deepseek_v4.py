# ---
# lambda-test: false
# ---

# # Serve DeepSeek V4 Pro on Modal with SGLang

# The DeepSeek V4 Pro weights are delivered in mixed MXFP4 and run through SGLang's
# `flashinfer_mxfp4` MoE runner backend on Blackwell, with EAGLE speculative decoding
# for low/moderate concurrency time-per-output-token.

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import aiohttp
import modal
import modal.experimental

here = Path(__file__).parent

# ## Set up the container image

# We use the `deepseek-v4-blackwell` tag of the SGLang image, which is the
# Blackwell-tuned build the SGLang team recommends for V4.

image = modal.Image.from_registry("lmsysorg/sglang:deepseek-v4-blackwell").entrypoint(
    []  # silence chatty logs on entry
)

# ### Load model weights

# We cache weights in a Modal
# [Volume](https://modal.com/docs/guide/volumes) and skip the download entirely
# when iterating with `dummy` weights.

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

image = image.env(
    {
        "HF_XET_HIGH_PERFORMANCE": "1",  # faster downloads
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    }
)


def download_model(repo_id, revision=None):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision)


REPO_ID = "deepseek-ai/DeepSeek-V4-Pro"

image = image.run_function(
    download_model,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    args=(REPO_ID,),
)

# ### Configure the inference engine

# We base the configuration of the engine on SGLang's
# [official cookbook recipe](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4).

# **Environment variables**

image = image.env(
    {
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_ENABLE_THINKING": "1",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    }
)

# You can also send any additional SGLang env vars set locally at deploy time.


def is_sglang_env_var(key):
    return key.startswith("SGL_") or key.startswith("SGLANG_")


image = image.env(
    {key: value for key, value in os.environ.items() if is_sglang_env_var(key)}
)

# **YAML**

default_config = """\
 # General Config
 host: 0.0.0.0
 log-level: debug  # very noisy

 # Model Config
 tool-call-parser: deepseekv4
 reasoning-parser: deepseek-v4
 trust-remote-code: true

 # Memory
 mem-fraction-static: 0.82
 chunked-prefill-size: 4096

 # MoE
 moe-runner-backend: flashinfer_mxfp4

 # Observability
 enable-metrics: true
 collect-tokens-histogram: true

 # Batching
 max-running-requests: 32
 cuda-graph-max-bs: 32

 # SpecDec (EAGLE, as recommended by the DeepSeek V4 release notes)
 speculative-algorithm: EAGLE
 speculative-num-steps: 3
 speculative-eagle-topk: 1
 speculative-num-draft-tokens: 4

 # Tuning
 disable-flashinfer-autotune: true
"""

local_config_path = os.environ.get("APP_LOCAL_CONFIG_PATH")

if modal.is_local():
    if local_config_path is None:
        local_config_path = here / "config_deepseek_v4.yaml"

        if not local_config_path.exists():
            local_config_path.write_text(default_config)

        print(
            f"Using default config from {local_config_path.relative_to(here)}:",
            default_config,
            sep="\n",
        )

    image = image.add_local_file(local_config_path, "/root/config.yaml")

# **Command-line arguments**


def _start_server() -> subprocess.Popen:
    """Start SGLang server in a subprocess"""
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(SGLANG_PORT),
        "--model-path",
        REPO_ID,
        "--tp-size",
        str(GPU_COUNT),
        "--config",
        "/root/config.yaml",
    ]

    print("Starting SGLang server with command:")
    print(*cmd)

    return subprocess.Popen(" ".join(cmd), shell=True, start_new_session=True)


with image.imports():
    import sglang  # noqa

# ## Configure infrastructure

app = modal.App("example-deepseek-v4", image=image)

# DeepSeek V4 Pro requires Blackwell for the MXFP4 MoE path and runs at TP=8,
# so we use eight B200s.

GPU_TYPE = "B200"
GPU_COUNT = 8

REGION = "us"
PROXY_REGIONS = ["us-east"]

MIN_CONTAINERS = 0  # Set to 1 for production to keep a warm replica

TARGET_INPUTS = 10  # Concurrent requests per replica before scaling

# ### Define the server

SGLANG_PORT = 8000
MINUTES = 60  # seconds
HOURS = 60 * MINUTES


@app.cls(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    scaledown_window=20 * MINUTES,
    timeout=3 * HOURS,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    region=REGION,
    min_containers=MIN_CONTAINERS,
)
@modal.experimental.http_server(
    port=SGLANG_PORT,
    proxy_regions=PROXY_REGIONS,
    exit_grace_period=25,
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class Server:
    @modal.enter()
    def start(self):
        """Start SGLang server process and wait for it to be ready"""
        self.proc = _start_server()
        wait_for_server_ready()

    @modal.exit()
    def stop(self):
        """Terminate the SGLang server process"""
        self.proc.terminate()
        self.proc.wait()


def wait_for_server_ready():
    """Wait for SGLang server to be ready"""
    import requests

    url = f"http://localhost:{SGLANG_PORT}/health"
    print(f"Waiting for server to be ready at {url}")

    while True:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print("Server is ready!")
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(5)


# ## Test the server

# ```bash
# modal run 06_gpu_and_ml/llm-serving/very_large_models_deepseek_v4.py
# ```


@app.local_entrypoint()
async def test(test_timeout=3 * HOURS, content=None, twice=True):
    """Test the model serving endpoint"""
    url = (await Server._experimental_get_flash_urls.aio())[0]

    system_prompt = {"role": "system", "content": "You are a helpful AI assistant."}

    if content is None:
        content = "Explain the transformer architecture in one paragraph."

    messages = [system_prompt, {"role": "user", "content": content}]

    print(f"Sending messages to {url}:", *messages, sep="\n\t")
    await probe(url, messages, timeout=test_timeout)

    if twice:
        messages[1]["content"] = "Write five different programs in Python and Rust."
        print(f"Sending second request to {url}:", *messages, sep="\n\t")
        await probe(url, messages, timeout=1 * MINUTES)


async def probe(url, messages, timeout=20 * MINUTES):
    """Send request with retry logic for startup delays"""
    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() < deadline:
            try:
                await _send_request_streaming(session, messages)
                return
            except asyncio.TimeoutError:
                await asyncio.sleep(1)
            except aiohttp.client_exceptions.ClientResponseError as e:
                if e.status == 503:  # Service Unavailable during startup
                    await asyncio.sleep(1)
                    continue
                raise e
    raise TimeoutError(f"No response from server within {timeout} seconds")


# ## Deploy the server

# ```bash
# modal deploy 06_gpu_and_ml/llm-serving/very_large_models_deepseek_v4.py
# ```


async def _send_request_streaming(
    session: aiohttp.ClientSession, messages: list, timeout: int | None = None
):
    """Stream response from chat completions endpoint"""
    payload = {
        "messages": messages,
        "stream": True,
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
            chunk = delta.get("content") or delta.get("reasoning_content")

            if chunk:
                print(
                    chunk,
                    end="",
                    flush="\n" in chunk or "." in chunk or len(chunk) > 100,
                )
                full_text += chunk
        print()
