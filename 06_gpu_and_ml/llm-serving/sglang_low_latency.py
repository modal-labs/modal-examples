# ---
# deploy: true
# cmd: ["python", "06_gpu_and_ml/llm-serving/sglang_low_latency.py"]
# ---

# # Low Latency Qwen 3-8B with SGLang and Modal

# In this example, we show how to serve [SGLang](https://github.com/sgl-project/sglang) at low latency on Modal.

# This example is intended to demonstrate everything required to run
# inference at the highest performance and with the lowest latency possible,
# and so it includes advanced features of both SGLang and Modal.
# For a simpler introduction to LLM serving, see
# [this example](https://modal.com/docs/examples/llm_inference).

# To minimize routing overheads, we use `@modal.experimental.http_server`,
# which uses a new, low latency routing service on Modal designed for latency-sensitive inference workloads.
# This gives us more control over routing, but with increased power comes increased responsibility.

# We also include instructions for cutting cold start times by an order of magnitude using Modal's [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/images).

# We start from a container image provided by the SGLang team via Dockerhub.

# While we're at it, we import the dependencies we'll need both remotely and locally (for deployment).

import asyncio
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60  # seconds

sglang_image: modal.Image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:v0.5.6.post2-cu129-amd64-runtime"
    ).entrypoint([])  # silence chatty logs on container start
)

# We also choose a GPU to deploy our inference server onto.
# We choose the H100 GPU, which offers excellent price-performance
# and supports 8bit floating point operations, which are the
# lowest precision well-supported in the relevant kernels
# across a variety of model architectures.

N_GPUS = 1
GPU = f"H100!:{N_GPUS}"

# ### Loading and cacheing the model weights

# We'll serve Alibaba's Qwen 3 LLM. For lower latency,
# we pick a smaller model (8B params)
# in a lower precision floating point format (FP8).
# This reduces the amount of data that needs to be loaded
# [from GPU RAM into SM SRAM](https://modal.com/gpu-glossary/perf/memory-bandwidth)
# in each forward pass.

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
MODEL_REVISION = (
    "220b46e3b2180893580a4454f21f22d3ebb187d3"  # latest commit as of 2026-01-01
)

# We load the model from the Hugging Face Hub, so we'll
# need their Python package.

sglang_image = sglang_image.uv_pip_install("huggingface-hub==0.36.0")

# We don't want to load the model from the Hub every time we start the server.
# We can load it much faster from a [Modal Volume](https://modal.com/docs/guide/volumes).
# Typical speeds are around one to two GB/s.

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
MODEL_PATH = f"{HF_CACHE_PATH}/{MODEL_NAME}"

# In addition to pointing the Hugging Face Hub at the path
# where we mount the Volume, we also turn on "high performance" downloads,
# which can fully saturate our GB/s bandwidth.

sglang_image = sglang_image.env(
    {"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"}
)

# ### Cacheing compilation artifacts

# Model weights aren't the only thing we want to cache.

# As a rule, LLM inference servers like SGLang don't directly provide their own kernels.
# They draw high-performance kernels from a variety of sources.

# As of version `0.5.6`, the default kernel backend
# for FP8 matrix multiplications (`fp8-gemm-backend`)
# on Hopper [SM architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
# GPUs like the H100 is
# [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
# by DeepSeek.

# The binaries of these kernels are not included in the SGLang Docker image and so
# must be [JIT-compiled](https://modal.com/gpu-glossary/host-software/nvrtc).
# We store these in a Modal Volume as well.

DG_CACHE_VOL = modal.Volume.from_name("deepgemm-cache", create_if_missing=True)
DG_CACHE_PATH = "/root/.cache/deepgemm"

# JIT DeepGEMM kernels are on by default, but we explicitly enable them via an environment variable.

sglang_image = sglang_image.env({"SGLANG_ENABLE_JIT_DEEPGEMM": "1"})

# We trigger the compilation by running `sglang.compile_deep_gemm` in a `subprocess`
# kicked off from a Python function.


def compile_deep_gemm():
    import os

    if int(os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1")):
        subprocess.run(
            f"python3 -m sglang.compile_deep_gemm --model-path {MODEL_NAME} --revision {MODEL_REVISION} --tp {N_GPUS}",
            shell=True,
        )


# We run this Python function on Modal as part of building the Image,
# so that it has access to the appropriate GPU and the caches for our model and compilaton artifacts.

sglang_image = sglang_image.run_function(
    compile_deep_gemm,
    volumes={DG_CACHE_PATH: DG_CACHE_VOL, HF_CACHE_PATH: HF_CACHE_VOL},
    gpu=GPU,
)

# ### Other environment variables

# Lastly, we set any additional environment variables.
# Here, we set a flag that improves compatibility of
# the [Torch Inductor compiler](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
# with GPU snapshotting.

sglang_image = sglang_image.env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})

# ## Speed up cold starts with GPU snapshotting

# ### Sleeping and waking an SGLang server

with sglang_image.imports():
    import requests


def sleep():
    requests.post(
        f"http://127.0.0.1:{PORT}/release_memory_occupation", json={}
    ).raise_for_status()


def wake_up():
    requests.post(
        f"http://127.0.0.1:{PORT}/resume_memory_occupation", json={}
    ).raise_for_status()


# ### Readiness checks and warmups for an SGLang server


def warmup():
    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=10
        ).raise_for_status()


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def wait_ready(process: subprocess.Popen, timeout: int = 5 * MINUTES):
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
            time.sleep(1)
    raise TimeoutError(f"SGLang server not ready within timeout of {timeout} seconds")


# ## Define the inference server and infrastructure

# ### Selecting infrastructure to minimize latency

# Minimizing latency requires geographic co-location of clients and servers.

# So for low latency LLM inference services on Modal, you must select a
# [cloud region](https://modal.com/docs/guide/region-selection)
# for both the GPU-accelerated containers running inference
# and for the Modal proxies that forward requests to them.

# Here, we assume users are mostly in the northern half of the Americas
# and select a single cloud region, `us-east`, to serve them.
# This should result in at most a few dozen milliseconds of round-trip time.

REGION = "us-east"

# For production-scale LLM inference services, there are generally
# enough requests to justify keeping at least one replica running at all times.
# Having a "warm" or "live" replica reduces latency by skipping slow initialization work
# that occurs when new replica boot up (a ["cold start"](https://modal.com/docs/guide/cold-start)).
# For LLM inference servers, that latency runs into the seconds or few tens of seconds,
# even with snapshotting.

# To ensure at least one container is always available,
# we can set the `min_containers` of our Modal Function
# to `1` or more.

# However, since this is sample code,

MIN_CONTAINERS = 0  # set to 1 to ensure one replica is always ready

app = modal.App(name="example-sglang-low-latency")

PORT = 8000

TARGET_INPUTS = 20


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL, DG_CACHE_PATH: DG_CACHE_VOL},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    region=REGION,
    min_containers=MIN_CONTAINERS,
)
@modal.experimental.http_server(
    port=PORT,
    proxy_regions=[REGION],
    exit_grace_period=5,  # seconds
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class SGLang:
    @modal.enter(snap=True)
    def startup(self) -> None:
        """Start the SGLang server and block until it is healthy."""

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
            "--cuda-graph-max-bs",  # only capture CUDA graphs for batch sizes we're likely to observe
            f"{TARGET_INPUTS * 2}",
            "--enable-metrics",
            "--enable-memory-saver",  # enable offload, for snapshotting
            "--enable-weights-cpu-backup",
        ]

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()
        sleep()

    @modal.enter(snap=False)
    def wake_up(self):
        wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()


## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy sglang_low_latency.py
# ```

# This will create a new app on Modal, build the container image for it if it hasn't been built yet,
# and deploy the app.

## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-sglang-low-latency-serve.us-east.modal.direct`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-sglang-low-latency-serve.us-east.modal.direct/docs`.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands.

# For simple routes like `/health`, which checks whether the server is responding,
# you can even send a request directly from the docs.

## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that does a healthcheck and then hits the server.

# If you execute the command

# ```bash
# modal run sglang_low_latency.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, prompt=None, twice=True):
    url = SGLang._experimental_get_flash_urls()[0]

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if prompt is None:
        prompt = "Explain the Singular Value Decomposition."

    content = [{"type": "text", "text": prompt}]

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    await probe(url, messages, timeout=test_timeout)
    if twice:
        messages[0]["content"] = "You are Jar Jar Binks."
        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await probe(url, messages, timeout=1 * MINUTES)


# This test relies on the two helper functions below,
# which ping the server and wait for a valid response.

# The `probe` helper function specifically ignores
# two types of errors that can occur while a replica
# is starting up -- timeouts on the client and 5XX responses
# from the server. Modal returns the [503 error code](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)
# to indicate that an `experimental.http_server` is not available.


async def probe(url, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [{"role": "user", "content": "Tell me a joke."}]

    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() < deadline:
            try:
                await _send_request(session, "llm", messages)
                return
            except (
                aiohttp.client_exceptions.ClientResponseError,
                asyncio.TimeoutError,
            ):
                await asyncio.sleep(1)
    raise TimeoutError(f"No response from server within {timeout} seconds")


async def _send_request(
    session: aiohttp.ClientSession,
    model: str,
    messages: list,
    timeout: int | None = None,
) -> None:
    async with session.post(
        "/v1/chat/completions",
        json={"messages": messages, "model": model},
        timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        print((await resp.json())["choices"][0]["message"]["content"])


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

# You can use the client code below to test the endpoint.
# It can be run with the command

# ```
# python sglang_low_latency.py
# ```


if __name__ == "__main__":
    # after deployment, we can use the class from anywhere
    sglang_server = modal.Cls.from_name("example-sglang-low-latency", "SGLang")

    print("calling inference server")
    try:
        asyncio.run(probe(sglang_server._experimental_get_flash_urls()[0]))
    except modal.exception.NotFoundError as e:
        raise Exception(
            f"To take advantage of GPU snapshots, deploy first with modal deploy {__file__}"
        ) from e
