# ---
# lambda-test: false
# ---

# # Serve Inkling-Small on Modal with SGLang

# [Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) is a natively
# multimodal Mixture-of-Experts model from Thinking Machines Lab. It accepts text, images
# and audio. Its decoder mixes sliding-window and full-attention layers, which keeps
# long-context inference affordable.

# We serve the NVFP4 checkpoint, quantized to four-bit floating point, so it needs a
# Blackwell GPU (B200 or newer) for the native FP4 tensor-core path.

# The engine flags below mirror the [recipe](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling)
# SGLang publishes. Several of them are required rather than tuning knobs; comments mark which.

# For more on serving large models efficiently, see the
# [high-performance LLM inference guide](https://modal.com/docs/guide/high-performance-llm-inference).
# For a gentler introduction to LLM serving on Modal, see
# [this example](https://modal.com/docs/examples/llm_inference).

import json
import subprocess
import time
import urllib.error
import urllib.request

import modal

# ## Set up the container image

# SGLang publishes model-launch image tags alongside its numbered releases, and Inkling
# has one: `inkling-cu13`. The generic `v0.5.x` tags do not work. They fail during
# warmup, because the FlashAttention-4 CuTe kernel this model requires is built against
# a different `nvidia-cutlass-dsl` than those images ship.

# `inkling-cu13` is a moving tag, so we pin the digest for reproducible builds. To use a
# newer build, look up the current digest with
# `curl -s https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/inkling-cu13`.

SGLANG_IMAGE = (
    "lmsysorg/sglang:inkling-cu13"
    "@sha256:5099f18aec336b30eb733524d2a277d1b348e4bf77a1c8950f90c96e6741f22e"
)

image = modal.Image.from_registry(SGLANG_IMAGE).entrypoint(
    []  # silence chatty logs on entry
)

# Draft-extend CUDA graph row width must include `draft_extend_num_front_tokens`,
# which this pinned image lacks, so we patch at image build.

image = image.run_commands(
    """python3 - <<'PY'
from pathlib import Path

path = Path("/sgl-workspace/sglang/python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py")
source = path.read_text()
old = "num_tokens_per_req=self.speculative_num_steps + 1,"
new = "num_tokens_per_req=self.speculative_num_steps + 1 + self.draft_extend_num_front_tokens,"
if source.count(old) != 1:
    raise RuntimeError("Unexpected SGLang source while backporting sglang#32254")
path.write_text(source.replace(old, new))
PY"""
)

# ### Load model weights

# We cache the weights in a Modal [Volume](https://modal.com/docs/guide/volumes) so that
# containers read them from Modal's filesystem instead of re-downloading on every cold
# start.

# Note the mount point. The SGLang image already ships files under
# `/root/.cache/huggingface`, and Modal will not mount a Volume over a non-empty path, so
# we put the cache at `/cache` and point `HF_HOME` there.

HF_CACHE_DIR = "/cache"
hf_cache_vol = modal.Volume.from_name("inkling-hf-cache", create_if_missing=True)

image = image.env(
    {
        "HF_HOME": HF_CACHE_DIR,
        "HF_XET_HIGH_PERFORMANCE": "1",  # faster downloads
    }
)

# Inkling's repositories require accepting the license, so downloading needs a Hugging
# Face token. Create the [Secret](https://modal.com/docs/guide/secrets) once with your own
# read token:

# ```
# modal secret create huggingface-secret HF_TOKEN=hf_...
# ```

hf_secret = modal.Secret.from_name("huggingface-secret")

REPO_ID = "thinkingmachines/Inkling-Small-NVFP4"


def download_model(repo_id, revision=None):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision, max_workers=16)


# We download as part of the image build, so `modal deploy` is the only command you need.
# It is roughly 171 GB, so the first build takes a while -- about ten minutes at Hugging
# Face's typical throughput. Later builds reuse the Volume and skip the download.

image = image.run_function(
    download_model,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    secrets=[hf_secret],
    args=(REPO_ID,),
    timeout=4 * 60 * 60,
    cpu=8,  # parallel shard downloads are CPU-bound on hashing
)

# ### Cache compiled kernels

# This model runs through `torch.compile` piecewise CUDA graphs, so a cold container
# spends most of its startup in TorchInductor codegen. Inductor writes to `/tmp` by
# default, which is ephemeral, so that cost is otherwise re-paid on every cold start.
# We point both caches at a second Volume, so only the first boot pays it.

# The cache is keyed to the shapes it compiled, so changing the GPU count or the
# speculation settings means compiling again.

COMPILE_CACHE_DIR = "/compile-cache"
compile_cache_vol = modal.Volume.from_name(
    "inkling-compile-cache", create_if_missing=True
)

image = image.env(
    {
        "TORCHINDUCTOR_CACHE_DIR": f"{COMPILE_CACHE_DIR}/inductor",
        "TRITON_CACHE_DIR": f"{COMPILE_CACHE_DIR}/triton",
        "SGLANG_CACHE_DIR": f"{COMPILE_CACHE_DIR}/sglang",
        # From the recipe's env block for this model.
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
        # Disable Inkling multi-stream overlap for stable CUDA-graph capture on
        # the MTP path at TP=1.
        "SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP": "0",
    }
)

# ### Configure the inference engine

# The checkpoint contains `mtp.safetensors` -- a multi-token-prediction head trained
# alongside the model. SGLang drives it through its EAGLE code path, so speculative
# decoding needs no additional model. The head has eight layers, which is where
# `--speculative-num-steps 8` comes from; the ninth draft token is the target model's own.

# It does cost memory, so the recipe lowers `--mem-fraction-static` when it is on.

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
        # Inkling asserts the attention backend is `fa4` or `triton`. SGLang's generic
        # default on Blackwell is `trtllm_mha`, which trips that assertion during
        # startup, so this must be set explicitly.
        "--attention-backend",
        "fa4",
        "--page-size",
        "128",
        "--fp4-gemm-backend",
        "flashinfer_trtllm",
        # Inkling's fused gate emits a packed top-k format, and `flashinfer_trtllm_routed`
        # is the runner that consumes it. The other NVFP4 MoE runners fail at warmup.
        "--moe-runner-backend",
        "flashinfer_trtllm_routed",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--mem-fraction-static",
        MEM_FRACTION_STATIC,
        # The sliding-window and short-convolution pools would otherwise size themselves
        # for the model's full context and crowd out everything else.
        "--swa-full-tokens-ratio",
        "0.1",
        "--mamba-full-memory-ratio",
        "0.1",
        "--enable-multimodal",
        # Inkling speaks in typed content blocks (`<|content_thinking|>`,
        # `<|content_invoke_tool_json|>`). Without these parsers, reasoning traces and
        # tool calls arrive as raw control tokens inside the message content.
        "--reasoning-parser",
        "inkling",
        "--tool-call-parser",
        "inkling",
        "--enable-metrics",
        # Skip SGLang's dummy warmup request so a failed first generate can't
        # kill the process before ready.
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
        ]

    # `--enable-torch-symm-mem` appears in the published recipe, which runs at TP=8. It
    # supports only world size 6 or 8 on this architecture.
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

# Cold starts are long: the weights are large, and a fresh container compiles kernels and
# captures CUDA graphs before it can serve a request. Budget an hour for the first boot
# and set `startup_timeout` accordingly. Boots of the *same* configuration are much
# quicker once the compile cache is warm.


@app.server(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    volumes={HF_CACHE_DIR: hf_cache_vol, COMPILE_CACHE_DIR: compile_cache_vol},
    secrets=[hf_secret],
    cpu=32,  # TorchInductor codegen at startup is CPU-bound and parallel
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

# The server speaks the OpenAI chat completions API, so any OpenAI-compatible client
# works. To spin up an ephemeral server and send it one request:

# ```
# modal run 06_gpu_and_ml/llm-serving/inkling_small.py
# ```

# One Inkling-specific detail: rather than a thinking on/off switch, its chat template
# takes a *reasoning effort* -- a name (`none`, `minimal`, `low`, `medium`, `high`,
# `max`) or a number from 0.0 to 0.99 -- passed through `chat_template_kwargs`. Because
# we enabled the `inkling` reasoning parser, the thinking trace arrives in
# `reasoning_content`, separate from the answer.


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

# Once you are happy with the configuration, deploy it so the endpoint stays up
# independently of your terminal:

# ```
# modal deploy 06_gpu_and_ml/llm-serving/inkling_small.py
# ```

# ## Addenda

# A few things worth knowing before you point real traffic at this.

# The endpoint above is `unauthenticated=True`, which is convenient for a demo and wrong
# for anything else. Add [proxy auth](https://modal.com/docs/guide/webhook-urls#authentication)
# before exposing it.

# `/health` returning 200 does not mean the server is fully warm. CUDA-graph capture
# covers the shapes it enumerated at startup, but a request whose shape was not captured
# still triggers kernel compilation on first touch. If you benchmark this, send a warmup
# pass before you start measuring, or the compile time lands inside your numbers.

# Turning MTP off is a one-line change (`ENABLE_MTP = False`), which also restores
# `--mem-fraction-static` to 0.85 since there is no draft model to make room for.
# Speculation pays most when you have latency headroom to spend; when the GPUs are
# already saturated by large batches, it buys much less.
