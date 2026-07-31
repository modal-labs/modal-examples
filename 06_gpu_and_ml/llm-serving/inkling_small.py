# ---
# lambda-test: false
# ---

# # Serve Inkling-Small on Modal with SGLang

# [Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) is a multimodal
# mixture-of-experts model from Thinking Machines Lab that accepts text, images, and audio.
# Its decoder combines sliding-window and full-attention layers to lower the cost of long
# context inference.

# This example serves the NVFP4 checkpoint which requires Blackwell GPUs (B200/B300s)
# to support its native four-bit tensor-core path.

# The engine flags follow SGLang's
# [recipe](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-Small#hw=b300&variant=default&quant=nvfp4&strategy=mtp&nodes=single).

# For more on serving large models efficiently, see the
# [high-performance LLM inference guide](https://modal.com/docs/guide/high-performance-llm-inference).
# For a simpler introduction to LLM serving on Modal, see
# [this example](https://modal.com/docs/examples/llm_inference).

import json
import subprocess
import time
import urllib.error
import urllib.request

import modal

# ## Set up the container image

SGLANG_IMAGE = (
    "lmsysorg/sglang:dev-inkling-dspark"
    "@sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7"
)

image = modal.Image.from_registry(SGLANG_IMAGE).entrypoint(
    []  # silence chatty logs on entry
)

# ### Load model weights

# Cache the weights in a Modal [Volume](https://modal.com/docs/guide/volumes)
# to avoid downloading them on every cold start.
# Note that the SGLang image already has files under `/root/.cache/huggingface`,
# so we mount the Volume at `/cache` and point `HF_HOME` there.

HF_CACHE_DIR = "/cache"
hf_cache_vol = modal.Volume.from_name("inkling-hf-cache", create_if_missing=True)

image = image.env(
    {
        "HF_HOME": HF_CACHE_DIR,
        "HF_XET_HIGH_PERFORMANCE": "1",  # faster downloads
    }
)

# Inkling's repositories require accepting the license, so downloading needs a Hugging
# Face token. Create the [Secret](https://modal.com/docs/guide/secrets) with:

# ```
# modal secret create huggingface-secret HF_TOKEN=hf_...
# ```

hf_secret = modal.Secret.from_name("huggingface-secret")

REPO_ID = "thinkingmachines/Inkling-Small-NVFP4"


def download_model(repo_id, revision=None):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision, max_workers=16)


image = image.run_function(
    download_model,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    secrets=[hf_secret],
    args=(REPO_ID,),
    timeout=4 * 60 * 60,
    cpu=8,  # parallel shard downloads are CPU-bound on hashing
)

# ### Cache compiled kernels


COMPILE_CACHE_DIR = "/compile-cache"
compile_cache_vol = modal.Volume.from_name(
    "inkling-compile-cache", create_if_missing=True
)

image = image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": f"{COMPILE_CACHE_DIR}/inductor",
        "TRITON_CACHE_DIR": f"{COMPILE_CACHE_DIR}/triton",
        "SGLANG_CACHE_DIR": f"{COMPILE_CACHE_DIR}/sglang",
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
    }
)

# ### Configure the inference engine

ENABLE_MTP = True
MEM_FRACTION_STATIC = "0.70" if ENABLE_MTP else "0.85"
MAX_TOTAL_TOKENS = 262_144

SGLANG_PORT = 8000
MINUTES = 60  # seconds
HOURS = 60 * MINUTES


def _server_command() -> list[str]:
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(SGLANG_PORT),
        "--model-path",
        REPO_ID,
        "--served-model-name",
        "inkling-small",
        "--trust-remote-code",
        "--tp",
        str(GPU_COUNT),
        "--quantization",
        "modelopt_fp4",
        "--attention-backend",
        "fa4",
        "--page-size",
        "128",
        "--fp4-gemm-backend",
        "flashinfer_trtllm",
        "--moe-runner-backend",
        "flashinfer_trtllm_routed",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--mem-fraction-static",
        MEM_FRACTION_STATIC,
        "--swa-full-tokens-ratio",
        "0.1",
        "--mamba-full-memory-ratio",
        "0.1",
        "--enable-multimodal",
        "--reasoning-parser",
        "inkling",
        "--tool-call-parser",
        "inkling",
        "--enable-metrics",
        # Skip SGLang's startup request. If that first generation fails, SGLang exits
        # before the server reports ready.
        "--skip-server-warmup",
    ]

    if ENABLE_MTP:
        cmd += [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "8",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "9",
            "--enable-multi-layer-eagle",
            "--speculative-use-rejection-sampling",
            "--max-total-tokens",
            str(MAX_TOTAL_TOKENS),
            "--max-running-requests",
            str(TARGET_INPUTS),
        ]

    if GPU_COUNT in (6, 8):
        cmd.append("--enable-torch-symm-mem")

    return cmd


# ## Configure infrastructure

# The NVFP4 weights are about 171 GB. A single B300 has 288 GB, which leaves enough
# memory for the model, target KV cache, and MTP draft pools.

GPU_TYPE = "B300"
GPU_COUNT = 1

MIN_CONTAINERS = 0  # set to 1 in production to keep a warm replica

TARGET_INPUTS = 32  # concurrent requests per replica before scaling out

app = modal.App("example-inkling-small", image=image)

# ### Define the server


@app.server(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={HF_CACHE_DIR: hf_cache_vol, COMPILE_CACHE_DIR: compile_cache_vol},
    secrets=[hf_secret],
    cpu=32,
    port=SGLANG_PORT,
    startup_timeout=1 * HOURS,
    scaledown_window=20 * MINUTES,
    exit_grace_period=25,
    min_containers=MIN_CONTAINERS,
    target_concurrency=TARGET_INPUTS,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        cmd = _server_command()
        print("starting SGLang with command:")
        print(" ".join(cmd))
        self.proc = subprocess.Popen(" ".join(cmd), shell=True, start_new_session=True)
        wait_for_server_ready(self.proc)

    @modal.exit()
    def stop(self):
        self.proc.terminate()
        self.proc.wait()


def is_server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def wait_for_server_ready(proc: subprocess.Popen):
    url = f"http://localhost:{SGLANG_PORT}/health"
    print(f"waiting for server to be ready at {url}")

    while True:
        # Surface a crashed engine immediately instead of waiting out startup_timeout.
        if proc.poll() is not None:
            raise RuntimeError(
                f"SGLang exited with code {proc.returncode} before becoming healthy"
            )
        if is_server_up(url):
            print("server is ready!")
            return
        time.sleep(5)


def wait_for_endpoint(url: str, timeout=1 * HOURS) -> None:
    deadline = time.monotonic() + timeout
    health = f"{url.rstrip('/')}/health"
    while True:
        if is_server_up(health):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the Server endpoint.")
        time.sleep(5)


# ## Test the server

# The server supports the OpenAI chat completions API.
# To spin up an ephemeral server and send it a request:

# ```
# modal run 06_gpu_and_ml/llm-serving/inkling_small.py
# ```

# Notably, Inkling's chat template takes a *reasoning effort* which can be set via a
# string (`none`, `minimal`, `low`, `medium`, `high`, `max`) or a number from 0.0 to 0.99.


@app.local_entrypoint()
def main(
    prompt: str = "In one sentence, why is sliding-window attention cheap?",
    reasoning_effort: str = "medium",
):
    url = Server.get_url()
    print(f"Server URL: {url}")
    wait_for_endpoint(url)

    payload = {
        "model": "inkling-small",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"sending a request to {url}")
    with urllib.request.urlopen(req, timeout=1 * HOURS) as resp:
        body = json.loads(resp.read())

    message = body["choices"][0]["message"]
    if message.get("reasoning_content"):
        print("--- reasoning ---")
        print(message["reasoning_content"])
    print("--- answer ---")
    print(message.get("content"))
    print("--- usage ---")
    print(body.get("usage"))


# ## Deploy the server

# ```
# modal deploy 06_gpu_and_ml/llm-serving/inkling_small.py
# ```

# ## Addenda

# For demonstration purposes, the endpoint is publicly accessible with `unauthenticated=True`. Add
# [proxy auth](https://modal.com/docs/guide/webhook-urls#authentication) before sending
# private data.

# Set `ENABLE_MTP = False` to disable speculation, which
# we recommend once large batches saturate the GPU.
