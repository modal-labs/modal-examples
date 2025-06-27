# # Run state-of-the-art RLMs on Blackwell GPUs with DeepSeek-R1-0528-FP4 and vLLM

# In this example, we demonstrate how to run NVIDIA's DeepSeek-R1-0528-FP4 model,
# a [state-of-the-art reasoning language model](https://lmarena.ai/leaderboard),
# by running a vLLM server in OpenAI-compatible mode on Modal's Blackwell GPUs.

# Our examples repository also includes scripts for running clients and load-testing for OpenAI-compatible APIs
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible).

# ## Set up the container image

# To run code on Modal, we define [container images](https://modal.com/docs/guide/images).
# All Modal containers have access to GPU drivers via the underlying host environment,
# and vLLM can be installed with `pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

# To take advantage of optimized kernels for CUDA 12.8, we install PyTorch, flashinfer, and their dependencies
# via an `extra` Python package index.

import json
import os
import shutil
import webbrowser
from pathlib import Path

import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "git clone -b feat/sharded-state-v1 --single-branch https://github.com/aarnphm/vllm.git",  # https://github.com/vllm-project/vllm/pull/19971
        "cd vllm && VLLM_USE_PRECOMPILED=1 uv pip install --system --compile-bytecode --editable .",  # https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source
    )
    .run_commands(
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.32.0 flashinfer-python==0.2.6.post1 --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "uv pip install --system --compile-bytecode torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    )
    .env(
        {
            "VLLM_LOGGING_LEVEL": "DEBUG",
        }
    )
)

# ## Configuring vLLM

# ### The V1 engine

# In its 0.7 release, in early 2025, vLLM added a new version of its backend infrastructure,
# the [V1 Engine](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).
# Using this new engine can lead to some [impressive speedups](https://github.com/modal-labs/modal-examples/pull/1064).
# It was made the default in version 0.8 and is [slated for complete removal by 0.11](https://github.com/vllm-project/vllm/issues/18571),
# in late summer of 2025.

# A small number of features, described in the RFC above, may still require the V0 engine prior to removal.
# Until deprecation, you can use it by setting the below environment variable to `0`.

vllm_image = vllm_image.env({"VLLM_USE_V1": "1"})

# ### Trading off fast boots and token generation performance

# vLLM has embraced dynamic and just-in-time compilation to eke out additional performance without having to write too many custom kernels,
# e.g. via the Torch compiler and CUDA graph capture.
# These compilation features incur latency at startup in exchange for lowered latency and higher throughput during generation.
# We make this trade-off controllable with the `FAST_BOOT` variable below.

FAST_BOOT = True

# If you're running an LLM service that frequently scales from 0 (frequent ["cold starts"](https://modal.com/docs/guide/cold-start))
# then you'll want to set this to `True`.

# If you're running an LLM service that usually has multiple replicas running, then set this to `False` for improved performance.

# See the code below for details on the parameters that `FAST_BOOT` controls.

# For more on the performance you can expect when serving your own LLMs, see
# [our LLM engine performance benchmarks](https://modal.com/llm-almanac).

# ## Download the model weights

# We'll be running a pretrained foundation model -- NVIDIA's DeepSeek-R1-0528-FP4

# Model parameters are often quantized to a lower precision during training
# than they are run at during inference.
# We'll use a four bit floating point quantization from NVIDIA.
# Native hardware support for FP4 formats in [Tensor Cores](https://modal.com/gpu-glossary/device-hardware/tensor-core)
# is limited to the latest [Streaming Multiprocessor architectures](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
# like those of Modal's [Hopper H100/H200 and Blackwell B200 GPUs](https://modal.com/blog/introducing-b200-h200).

MODEL_NAME = "nvidia/DeepSeek-R1-0528-FP4"
MODEL_REVISION = "91cfc7c35acd8ecfc769205989310208b8b81c9c"

# Although vLLM will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use [Modal Volumes](https://modal.com/docs/guide/volumes) for our cache.
# Modal Volumes are essentially a "shared disk" that all Modal Functions can access like it's a regular disk.

app_name = "example-vllm-deepseek"

hf_cache_vol = modal.Volume.from_name(f"{app_name}-hf-cache", create_if_missing=True)

# We'll also cache some of vLLM's JIT compilation artifacts in a Modal Volume.

vllm_cache_vol = modal.Volume.from_name(
    f"{app_name}-vllm-cache", create_if_missing=True
)

# We set the `HF_HOME` and `VLLM_CACHE_ROOT` environment variables to point to the Volumes so that the model weights
# and JIT compilation artifacts are cached there.

HF_CACHE_PATH = Path("/hf_cache")
VLLM_CACHE_PATH = Path("/vllm_cache")

MODELS_PATH = HF_CACHE_PATH / "models"
ARTIFACTS_PATH = VLLM_CACHE_PATH / "artifacts"

volumes = {
    HF_CACHE_PATH: hf_cache_vol,
    VLLM_CACHE_PATH: vllm_cache_vol,
}

# Note though that even though we've pre-download the model weights,
# having all workers (1 per GPU) read the entire checkpoint takes a while.
# Therefore, we'll [shard the checkpoint](https://docs.vllm.ai/en/stable/examples/offline_inference/save_sharded_state.html)
# such that each worker loads their model state dict separately.

N_GPUS = 8
GPU_CONFIG = f"B200:{N_GPUS}"
MODEL_DIR = MODELS_PATH / MODEL_NAME
MODEL_SHARD_DIR = MODEL_DIR / "sharded"


def download_model():
    from huggingface_hub import snapshot_download

    print("downloading base model if necessary")
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors
        revision=MODEL_REVISION,
    )


def shard_model():
    from vllm import LLM

    print("sharding checkpoint for each GPU")
    llm = LLM(
        model=str(MODEL_DIR),
        tensor_parallel_size=N_GPUS,
        revision=MODEL_REVISION,
    )

    Path(MODEL_SHARD_DIR).mkdir(exist_ok=True)

    # Check which engine version is being used
    is_v1_engine = hasattr(llm.llm_engine, "engine_core")

    if is_v1_engine:
        # For V1 engine, we need to use engine_core.save_sharded_state
        print("Using V1 engine save path")
        llm.llm_engine.engine_core.save_sharded_state(
            path=str(MODEL_SHARD_DIR),
        )
    else:
        # For V0 engine
        print("Using V0 engine save path")
        model_executor = llm.llm_engine.model_executor
        model_executor.save_sharded_state(
            path=str(MODEL_SHARD_DIR),
        )

    # Copy metadata files to output directory
    for file in os.listdir(MODEL_DIR):
        if os.path.splitext(file)[1] not in (".bin", ".pt", ".safetensors"):
            if os.path.isdir(os.path.join(MODEL_DIR, file)):
                shutil.copytree(
                    os.path.join(MODEL_DIR, file),
                    os.path.join(MODEL_SHARD_DIR, file),
                    dirs_exist_ok=True,
                )
            else:
                shutil.copy(os.path.join(MODEL_DIR, file), MODEL_SHARD_DIR)


MINUTES = 60  # seconds
vllm_image = (
    vllm_image.env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model transfers
            "HF_HOME": str(MODELS_PATH),
            "VLLM_CACHE_ROOT": str(ARTIFACTS_PATH),
        }
    )
    .run_function(download_model, volumes=volumes, timeout=40 * MINUTES)
    .run_function(shard_model, volumes=volumes, timeout=60 * MINUTES, gpu=GPU_CONFIG)
)

# On the first container start, we mount the Volume, download the model, and build the engine,
# which takes a few minutes. Subsequent starts will be much faster,
# as the engine is cached in the Volume and loaded in seconds.

# Container starts are triggered when Modal scales up your Function,
# like the first time you run this code or the first time a request comes in after a period of inactivity.
# For details on optimizing container start latency, see
# [this guide](https://modal.com/docs/guide/cold-start).

# ## Build a vLLM engine and serve it

# The function below spawns a vLLM instance listening at port 8000, serving requests to our model.
# We wrap it in the [`@modal.web_server` decorator](https://modal.com/docs/guide/webhooks#non-asgi-web-servers)
# to connect it to the Internet.

# The server runs in an independent process, via `subprocess.Popen`, and only starts accepting requests
# once the model is spun up and the `serve` function returns.

app = modal.App(app_name)

MAX_BATCH_SIZE = 32  # how many requests can one replica handle? tune carefully!
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=40 * MINUTES,  # how long should we wait for container start?
    volumes=volumes,
)
@modal.concurrent(max_inputs=MAX_BATCH_SIZE)
@modal.web_server(port=VLLM_PORT, startup_timeout=40 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        str(MODEL_SHARD_DIR),
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--load-format",
        "sharded_state",
        "--model-loader-extra-config",
        '{"strict": false}',
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPUS)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy vllm_deepseek.py
# ```

# This will create a new app on Modal, build the container image for it if it hasn't been built yet,
# and deploy the app.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-vllm-deepseek-serve.modal.run`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-vllm-openai-compatible-serve.modal.run/docs`.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands.

# For simple routes like `/health`, which checks whether the server is responding,
# you can even send a request directly from the docs.

# To interact with the API programmatically in Python, we recommend the `openai` library.

# See the `client.py` script in the examples repository
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible)
# to take it for a spin:

# ```bash
# # pip install openai==1.76.0
# python openai_compatible/client.py
# ```

# ## Testing the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that does a healthcheck and then hits the server.

# If you execute the command

# ```bash
# modal run vllm_deepseek.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=40 * MINUTES):
    url = serve.get_web_url()

    system_prompt = {
        "role": "system",
        "content": """
        You are a helpful assistant that only responds in English and can generate HTML pages, and only responds with the HTML.
        """,
    }

    prompt = """
    Create an HTML page implementing a simple game of tic-tac-toe.
    Only output the HTML, no other text.
    """

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": prompt},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        output = await _send_request(session, "llm", messages)

    # post-process
    html_content = (
        output.split("</think>")[-1].split("```html")[-1].split("```")[0].strip()
    )

    html_filename = Path(__file__).parent / "tic_tac_toe.html"
    with open(html_filename, "w") as f:
        f.write(html_content)

    file_path = os.path.abspath(html_filename)
    file_url = f"file://{file_path}"
    print(f"\nHTML saved to: {file_path}")
    print(f"Opening in browser: {file_url}")
    webbrowser.open(file_url)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> str:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, object] = {"messages": messages, "model": model, "stream": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    output = ""
    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=1 * MINUTES
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            # extract new content and stream it
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):  # SSE prefix
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert (
                chunk["object"] == "chat.completion.chunk"
            )  # or something went horribly wrong
            content = chunk["choices"][0]["delta"]["content"]
            if content:
                output += content
                print(content, end="")
    print()
    return output
