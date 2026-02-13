# ---
# cmd: ["modal", "run", "06_gpu_and_ml/llm-serving/very_large_models.py"]
# env: {"APP_USE_DUMMY_WEIGHTS": "1"}
# ---

# # Serve very large language models (DeepSeek V3, Kimi-K2, GLM 4.7/5)

# This example demonstrates the basic patterns for serving language models on Modal
# whose weights consume hundreds of gigabytes of storage.

# In short:

# - load weights into a Modal Volume ahead of server launch
# - use random "dummy" weights when iteratively developing your server
# - use two, four, or eight H200 or B200 GPUs
# - use lower-precision weight formats (FP4 on Blackwell, FP8 on Hopper)
# - default to using speculative decoding, especially if batches are in the few tens of sequences

# For more tips on how to serve specific types of LLM inference at high performance,
# see [this guide](https://modal.com/docs/guide/high-performance-llm-inference).
# For a gentler introduction to LLM serving,
# see [this example](https://modal.com/docs/examples/llm_inference).

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

# We start by creating a Modal Image based on the Docker image
# provided by the SGLang team.
# This contains our Python and system dependencies.
# Add more by chaining `.apt_install` and `.uv_pip_install`
# or `.pip_install`  method calls, as we do below with
# `.entrypoint`.
# See the [Modal Image guide](https://modal.com/docs/guide/images)
# for details.

image = modal.Image.from_registry("lmsysorg/sglang:v0.5.7").entrypoint(
    []  # silence chatty logs on entry
)

# ### Load model weights

# Large model weights take a long time to move around.
# Model weight servers like Hugging Face will send weights
# at a few hundred megabytes per second. For large models,
# with weight sizes in the hundreds of gigabytes,
# that means thousands of seconds (tens of minutes)
# of model loading time.

# After loading them we can cache these weights in a Modal
# [Volume](https://modal.com/docs/guide/volumes)
# so that they are loaded about 10x faster --
# about one to three gigabytes per second.

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# That still means minutes of startup time.
# Both of these latencies kill productivity when you're iterating
# on aspects besides model behavior, like server configuration.

# For this reason, we recommend skipping model loading while you're developing
# a server or configuration -- even when benchmarking, if you can!
# You can still exercise the same code paths if you use the `dummy` model
# loading format. In this sample code, we add an `APP_USE_DUMMY_WEIGHTS` environment variable
# to control this behavior from the command line during iteration.

USE_DUMMY_WEIGHTS = os.environ.get("APP_USE_DUMMY_WEIGHTS", "0").lower() in (
    "1",
    "true",
)

image = image.env(
    {
        "HF_XET_HIGH_PERFORMANCE": "1",  # faster downloads
        "APP_USE_DUMMY_WEIGHTS": str(int(USE_DUMMY_WEIGHTS)),
    }
)

# We download the model weights from Hugging Face by
# running a Python function as part of the Modal Image build.
# Note that command-line logging will be somewhat limited.


def download_model(repo_id, revision=None):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision)


# To run the function, we need to pick a specific model to download.
# We'll use Z.ai's GLM 4.7 in eight bit
# [floating point quantization](https://quant.exposed).
# This model takes about thirty minutes to an hour to download from Hugging Face.

REPO_ID = "zai-org/GLM-4.7-FP8"

if not USE_DUMMY_WEIGHTS:  # skip download if we don't need real weights
    image = image.run_function(
        download_model,
        volumes={"/root/.cache/huggingface": hf_cache_vol},
        args=(REPO_ID,),
    )

# ### Configure the inference engine

# Running large models efficiently requires specialized inference engines like SGLang.
# These engines are generally highly configurable.

# For SGLang, there are three main sources of configuration values:

# - _Environment variables_ for the process running `sglang`.
# - _Command-line arguments_ for the command to launch the `sglang` process.
# - _Configuration files_ loaded by the `sglang` process.

# For deployments, we prefer to put information in configuration files where possible.
# CLI arguments and configuration files can typically be interchanged.
# CLI arguments are convenient when iterating, but configuration files are easier to share.
# We use environment variables only as a last resort, typically to activate new or experimental features.

# **Environment variables**

# SGLang environment variables are prefixed with `SGL_` or `SGLANG_`.
# The `SGL_` prefix is deprecated.

# The snippet below adds any such environment variables
# present during deployment to the Modal Image.


def is_sglang_env_var(key):
    return key.startswith("SGL_") or key.startswith("SGLANG_")


image = image.env(
    {key: value for key, value in os.environ.items() if is_sglang_env_var(key)}
)

# **YAML**

# Configuration files can be passed in YAML format.

# We include a default config in-line in the code here for ease of use.
# It's designed to run GLM 4.7 FP8 at low to moderate concurrency.
# In particular, it uses that model's built-in multi-token prediction speculative decoding to improve
# [time per output token](https://modal.com/llm-almanac/how-to-benchmark).

default_config = """\
 # General Config
 host: 0.0.0.0
 log-level: debug  # very noisy

 # Model Config
 tool-call-parser: glm47
 reasoning-parser: glm45
 trust-remote-code: true

 # Memory
 mem-fraction-static: 0.85
 chunked-prefill-size: 32768
 kv-cache-dtype: fp8_e4m3

 # Observability
 enable-metrics: true
 collect-tokens-histogram: true

 # Batching
 max-running-requests: 32
 cuda-graph-max-bs: 32

 # SpecDec (speed up low/moderate concurrency)
 speculative-algorithm: EAGLE  # built into GLM 4.7, is just multi-token prediction
"""

# You'll want to provide your own configuration file for other settings,
# in particular if you change the model.

# We add an environment variable, `APP_LOCAL_CONFIG_PATH`,
# to change the loaded configuration.

local_config_path = os.environ.get("APP_LOCAL_CONFIG_PATH")

if modal.is_local():
    if local_config_path is None:
        local_config_path = here / "config_very_large_models.yaml"

        if not local_config_path.exists():
            local_config_path.write_text(default_config)

        print(
            f"Using default config from {local_config_path.relative_to(here)}:",
            default_config,
            sep="\n",
        )

    image = image.add_local_file(local_config_path, "/root/config.yaml")

# ** Command-line arguments**

# We launch our server by kicking off a subprocess.
# The convenience function below encapsulates the command
# and its arguments.

# We pass a few key bits of configuration that are consumed
# by other code here, rather than in a configuration file,
# so that values stay in sync.

# That includes:

# - Model information, which is also used during weight cacheing
# - GPU count, which is also used below when defining our Modal deployment
# - the port to serve on, which is also used to connect up Modal networking

# We also pass the `HF_HUB_OFFLINE` environment variable here,
# so that our server will crash when trying to load the real model
# if those weights are not in cache.
# For smaller models, we can instead load weights dynamically on
# server start (and cache them so later starts are faster).
# But for large models, weight loading extends the first start latency
# so much that downstream timeouts are triggered --
# or need to be extended so much that they are no longer tight enough
# on the happy path.


def _start_server() -> subprocess.Popen:
    """Start SGLang server in a subprocess"""
    cmd = [
        f"HF_HUB_OFFLINE={0 if USE_DUMMY_WEIGHTS else 1}",
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

    if USE_DUMMY_WEIGHTS:
        cmd.extend(["--load-format", "dummy"])

    print("Starting SGLang server with command:")
    print(*cmd)

    return subprocess.Popen(" ".join(cmd), shell=True, start_new_session=True)


# Lastly, we import the `sglang` library as part of loading the Image on Modal.
# This is a minor optimization, but it can shave a few seconds off cold start latencies
# by providing better prefetching hints, and every second counts!

with image.imports():
    import sglang  # noqa

# ## Configure infrastructure

# Now, we wrap our configured SGLang server for our large model
# in the infrastructure required to run and interact with it.
# Infrastrucure in Modal is generally attached to an App.
# Here, we'll attach our Modal Image as the default for
# Modal Functions that run in the App.

app = modal.App("example-serve-very-large-models", image=image)

# Most importantly, we need to decide what hardware to run on.
# [H200 and B200 GPUs](https://modal.com/blog/introducting-b200-h200)
# have over 100 GB of [GPU RAM](https://modal.com/gpu-glossary/device-hardware/gpu-ram) --
# 141 GB and 180 GB, respectively.
# The model's weights will be stored in this memory,
# and they consume several hundred gigabytes of space,
# so we will generally want several of these accelerators.
# We also need space for the model's KV cache of activations
# on input sequences.

# In eight-bit precision, GLM 4.7 consumes ~350 GB of space,
# so we use four H200s for 564 GB of RAM.

GPU_TYPE = "H200"
GPU_COUNT = 4

# We'll use a Modal `experimental.http_server` to serve our model.
# This reduces client latencies and provides for regionalized deployment.
# You can read more about it in [this example](https://modal.com/docs/examples/sglang_low_latency).
# To configure it, we need to pass in region information for the GPU workers
# and for the load-balancing proxy.

REGION = "us"
PROXY_REGIONS = ["us-east"]

# Lastly, we need to configure autoscaling parameters.
# By default, Modal is fully serverless, and applications
# scale to zero when there is no load.
# But booting up inference engines for large models takes minutes,
# which is generally longer than clients can tolerate waiting.

# So a production deployment of large models that has clients with
# per-request SLAs in the few or tens of seconds
# generally needs to keep one replica up at all times.
# In Modal, we achieve this with the `min_containers` parameter
# of `App.cls` or `App.function`.

# This can trigger substantial costs, so we leave the value at `0`
# in this sample code.

MIN_CONTAINERS = 0  # Set to 1 for production to keep a warm replica

# Deployments of large models with a single node per replica can generally handle a few tens of requests
# without queueing. When a particular replica has more requests than it can handle, we want to scale it up.
# This behavior is configured by passing the `target_inputs` parameter to `modal.concurrent`.

TARGET_INPUTS = 10  # Concurrent requests per replica before scaling

# ### Define the server

# Now we're ready to put all of our infrastructure configuration
# together into a Modal Cls.

# The Modal Cls allows us to control
# [container lifecycle](https://modal.com/docs/guide/lifecycle-functions).
# In particular, it lets us define work that a replica should do before
# and after it handles requests in methods decorated with `modal.enter`
# and `modal.exit`, respectively.

SGLANG_PORT = 8000
MINUTES = 60  # seconds


@app.cls(
    image=image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    scaledown_window=20 * MINUTES,  # how long should we stay up with no requests?
    timeout=30 * MINUTES,  # how long should we wait for container start?
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    region=REGION,
    min_containers=MIN_CONTAINERS,
)
@modal.experimental.http_server(
    port=SGLANG_PORT,
    proxy_regions=["us-east"],
    exit_grace_period=25,  # time to finish requests on shutdown (seconds)
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


# We called a `wait_for_server_ready` function in our `modal.enter` method.
# That's defined below. It pings the `/health` endpoint until the server responds.


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

# You can deploy a fresh replica and test it
# using the command

# ```bash
# APP_USE_DUMMY_WEIGHTS=1 modal run very_large_models.py
# ```

# which will create an ephemeral Modal App
# and execute the `local_entrypoint` code below.

# Because the weights are randomized, the outputs are also random.
# Remove the `APP_USE_DUMMY_WEIGHTS` flag to test the trained model.


@app.local_entrypoint()
async def test(test_timeout=20 * MINUTES, content=None, twice=True):
    """Test the model serving endpoint"""
    url = Server._experimental_get_flash_urls()[0]

    if USE_DUMMY_WEIGHTS:
        system_prompt = {"role": "system", "content": "This system produces gibberish."}
    else:
        system_prompt = {"role": "system", "content": "You are a helpful AI assistant."}

    if content is None:
        content = "Explain the transformer architecture in one paragraph."

    messages = [system_prompt, {"role": "user", "content": content}]

    print(f"Sending messages to {url}:", *messages, sep="\n\t")
    await probe(url, messages, timeout=test_timeout)

    if twice:
        messages[1]["content"] = "What is the capital of France?"
        print(f"Sending second request to {url}:", *messages, sep="\n\t")
        await probe(url, messages, timeout=1 * MINUTES)


# The unique client logic for Modal deployments is in the `probe` function below.
# Specifically, when a Modal `experimental.http_server` is spinning up,
# i.e. before the `modal.enter` finishes for at least one replica,
# clients will see a `503 Service Unavailable` status
# and so should retry.


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

# When you're ready, you can create a persistent deployment with

# ```bash
# APP_USE_DUMMY_WEIGHTS=0 modal deploy very_large_models.py
# ```

# And hit it with any OpenAI API-compatible client!

# ## Addenda

# The `probe` function above uses this helper function
# to stream response tokens as they become available.


async def _send_request_streaming(
    session: aiohttp.ClientSession, messages: list, timeout: int | None = None
):
    """Stream response from chat completions endpoint"""
    payload = {
        "messages": messages,
        "stream": True,
        "max_tokens": 1024 if USE_DUMMY_WEIGHTS else None,
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
