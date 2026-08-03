# # Deploy DeepSeek-V4-Flash with SGLang and Modal

# We'll show in this example how to serve
# [DeepSeek-V4-Flash](https://arxiv.org/abs/2606.19348), a Mixture-of-Experts (MoE)
# model with 284B total parameters and 13B active.

# It achieves comparable reasoning performance to its bigger variant,
# the [DeepSeek-V4-Pro preview](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), while being much more compact in terms of
# model parameters.

# ## Set up the container image

# An issue currently exists with the drafter incorrectly rewriting states.
# While not yet merged, we apply the fix in this
# [open PR](https://github.com/sgl-project/sglang/pull/32183) manually
# to the container image provided by the SGLang team.

import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request

import modal

MINUTES = 60  # seconds
GB = 1024  # mb

PR32183_DIFF_URL = (
    "https://github.com/sgl-project/sglang/compare/"
    "5387e23ecd7dde4c383ae857983686e6a73bddf3..."
    "22ef431215b1d8529eaebd8e8c6de9510390afaf.diff"
)
PR32183_DIFF_SHA256 = "ddd65902ba570c158f9d6783604cf7d9f2f13bf41994fcbf330a68ea1909923c"

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu13-20260729-16a52bff")
    .entrypoint([])  # silence chatty logs on container start
    .run_commands(
        f"curl -fsSL {PR32183_DIFF_URL} -o /tmp/pr32183.diff",
        f"echo '{PR32183_DIFF_SHA256}  /tmp/pr32183.diff' | sha256sum -c -",
        "cd /sgl-workspace/sglang"
        " && git apply --stat --exclude=test/* /tmp/pr32183.diff"
        " && git apply --exclude=test/* /tmp/pr32183.diff",
    )
)

# ### Load and cache the model weights and kernels

# Downloads from the Hugging Face Hub are much faster if you are authenticated,
# so we add a Hugging Face token as a [Modal Secret](https://modal.com/docs/guide/secrets) with:

# ```
# modal secret create huggingface-secret HF_TOKEN=hf_...
# ```

MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"

hf_secret = modal.Secret.from_name("huggingface-secret")

# We don't want to load the model from the Hub every time we start the server.
# So instead, we load the cached weights from a [Modal Volume](https://modal.com/docs/guide/volumes).
# Note that the container image already contains files at the default location
# `/root/.cache/huggingface`, so we specify a different path.

HF_CACHE_DIR = "/cache/huggingface"
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# We also want to turn on
# [high performance downloads](https://huggingface.co/docs/hub/en/models-downloading#faster-downloads)
# to fully saturate our network bandwidth.

sglang_image = sglang_image.env(
    {"HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub", "HF_XET_HIGH_PERFORMANCE": "1"}
)


def download_model(repo_id, revision=None):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision, max_workers=16)


sglang_image = sglang_image.run_function(
    download_model,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    secrets=[hf_secret],
    args=(MODEL_NAME, MODEL_REVISION),
    timeout=4 * 60 * MINUTES,
    cpu=8,
)

# As part of the loading process, the model compiles DeepGEMM and FlashInfer kernels.
# To avoid recompilation on cold-starts, we specify a path to a Volume
# for the compiled kernels to live in.

DG_CACHE_DIR = "/cache/deep_gemm"
FLASHINFER_CACHE_DIR = "/root/.cache/sglang/flashinfer"

dg_cache_vol = modal.Volume.from_name("sglang-deepgemm-cache", create_if_missing=True)
flashinfer_cache_vol = modal.Volume.from_name(
    "flashinfer-autotune-cache", create_if_missing=True
)

sglang_image = sglang_image.env(
    {
        "SGLANG_DG_CACHE_DIR": DG_CACHE_DIR,
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
        "TILELANG_CACHE_DIR": f"{DG_CACHE_DIR}/tilelang",
    }
)

# ## Configure the infrastructure

# We choose a [GPU](https://modal.com/docs/guide/gpu) to deploy our inference server onto.
# Conveniently, a single B300 can hold the model weights, KV cache, and speculative decoding module.
# It offers excellent price-performance and supports both 8 bit and 4 bit
# [quantized floating point](https://modal.com/llm-almanac/quant-formats) operations.

GPU_TYPE, GPU_COUNT = "B300", 1
CPU = 8
MEMORY = 96 * GB

# For production-scale LLM inference services, there are generally
# enough requests to justify keeping at least one replica running at all times.
# However, to ensure at least one container is always available,
# we can set `min_containers` to `1` or more for our inference server.

MIN_CONTAINERS = 0  # set to 1 in production to keep a warm replica

# Modal empowers you to decide how to scale up and down replicas
# in response to load. Without autoscaling, users' requests will queue
# when the server becomes overloaded or simply face higher latencies
# once above a certain minimum number of concurrent requests.

TARGET_INPUTS = 24

# Modal considers a new replica ready to receive inputs once the
# [`modal.enter`](https://modal.com/docs/guide/lifecycle-functions)
# methods have exited and the container accepts connections.
# To ensure that our server is actually ready for inputs,
# we define helper functions to check and ensure the server is ready
# from both within the container and a local client.

STARTUP_TIMEOUT = 60 * MINUTES


def is_server_up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def wait_ready(proc: subprocess.Popen):
    url = f"http://localhost:{DEFAULT_PORT}/health"
    print(f"waiting for server to be ready at {url}")

    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"SGLang exited with code {proc.returncode} before becoming healthy"
            )
        if is_server_up(url):
            print("server is ready!")
            return
        time.sleep(5)


def warmup():
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        req = urllib.request.Request(
            f"http://localhost:{DEFAULT_PORT}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5 * MINUTES) as resp:
                resp.read()
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(f"warmup request failed, continuing: {exc}")


def wait_for_endpoint(url: str, timeout: int = STARTUP_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    health = f"{url.rstrip('/')}/health"
    while True:
        if is_server_up(health):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the Server endpoint.")
        time.sleep(5)


# ## Define the inference server

# For maximum performance, we set a few bespoke enviroment variables
# and engine flags.

sglang_image = sglang_image.env(
    {
        "NCCL_CUMEM_ENABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_DEFAULT_THINKING": "false",
        "SGLANG_TIMEOUT_KEEP_ALIVE": f"{5 * MINUTES}",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
)

DEFAULT_PORT = 8000


def _server_command() -> list[str]:
    cmd = [
        "sglang",
        "serve",
        "--model-path",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(DEFAULT_PORT),
        "--tp",
        str(GPU_COUNT),
        "--chunked-prefill-size",
        "4096",
        "--context-length",
        "268000",
        "--cuda-graph-max-bs-decode",
        "64",
        "--decode-log-interval",
        "200",
        "--default-chat-template-kwargs",
        '{"thinking":false}',
        "--disable-flashinfer-autotune",
        "--dist-timeout",
        f"{60 * MINUTES}",
        "--max-running-requests",
        "64",
        "--mem-fraction-static",
        "0.90",
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--reasoning-parser",
        "deepseek-v4",
        "--speculative-algorithm",
        "DSPARK",
        "--swa-full-tokens-ratio",
        "0.1",
        "--tool-call-parser",
        "deepseekv4",
        "--trust-remote-code",
        "--skip-server-warmup",
    ]
    return cmd


# Onto the main event that is defining our inference server.

app = modal.App(name="example-deepseek-v4-flash")


@app.server(
    image=sglang_image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        DG_CACHE_DIR: dg_cache_vol,
        FLASHINFER_CACHE_DIR: flashinfer_cache_vol,
    },
    cpu=CPU,
    memory=MEMORY,
    port=DEFAULT_PORT,
    startup_timeout=STARTUP_TIMEOUT,
    exit_grace_period=25,  # seconds, time to finish up requests when closing down
    min_containers=MIN_CONTAINERS,
    target_concurrency=TARGET_INPUTS,
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def startup(self):
        cmd = _server_command()
        print(shlex.join(cmd))
        self.proc = subprocess.Popen(cmd, start_new_session=True)
        wait_ready(self.proc)
        warmup()

    @modal.exit()
    def stop(self):
        self.proc.terminate()
        self.proc.wait()


# ## Deploy the server

# To deploy the server on Modal, just run

# ```bash
# modal deploy 06_gpu_and_ml/llm-serving/deepseek_v4_flash.py
# ```

# This will create a new App on Modal and build the container image for it if it hasn't been built yet.

# ## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that hits the server with a simple client.

# If you execute the command

# ```bash
# modal run 06_gpu_and_ml/llm-serving/deepseek_v4_flash.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# This is akin to running simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
def main(
    prompt: str = "Explain why tech bros and climbers are an increasing phenomenon.",
):
    url = Server.get_url()
    print(f"server url: {url}")
    wait_for_endpoint(url)

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    print(f"sending a request to {url}")
    with urllib.request.urlopen(req, timeout=STARTUP_TIMEOUT) as resp:
        body = json.loads(resp.read())

    message = body["choices"][0]["message"]
    print(message.get("content"))
    print(body.get("usage"))
