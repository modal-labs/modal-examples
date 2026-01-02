# ---
# deploy: true
# cmd: ["python", "06_gpu_and_ml/llm-serving/sglang_low_latency.py"]
# ---

# # Low latency Qwen 3-8B with SGLang and Modal

# In this example, we show how to serve [SGLang](https://github.com/sgl-project/sglang) at low latency on Modal.

# This example is intended to demonstrate everything required to run
# inference at the highest performance and with the lowest latency possible,
# and so it includes advanced features of both SGLang and Modal.
# For a simpler introduction to LLM serving, see
# [this example](https://modal.com/docs/examples/llm_inference).

# To minimize routing overheads, we use `@modal.experimental.http_server`,
# which uses a new, low-latency routing service on Modal designed for latency-sensitive inference workloads.
# This gives us more control over routing, but with increased power comes increased responsibility.

# We also include instructions for cutting cold start times by an order of magnitude using Modal's
# [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/images).

# We start from a container image provided
# [by the SGLang team via Dockerhub](https://hub.docker.com/r/lmsysorg/sglang/tags).

# While we're at it, we import the dependencies we'll need both remotely and locally (for deployment).

import asyncio
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60  # seconds

sglang_image = (
    modal.Image.from_registry(
        "lmsysorg/sglang:v0.5.6.post2-cu129-amd64-runtime"
    ).entrypoint([])  # silence chatty logs on container start
)

# We also choose a GPU to deploy our inference server onto.
# We choose the [H100 GPU](https://modal.com/blog/introducing-h100),
# which offers excellent price-performance
# and supports 8bit floating point operations, which are the
# lowest precision well-supported in the relevant [GPU kernels](https://modal.com/gpu-glossary/device-software/kernel)
# across a variety of model architectures.

# We run on two of them to double our effective [memory bandwidth](https://modal.com/gpu-glossary/perf/memory-bandwidth)
# during tensor parallel operations like large matrix multiplications.
# This leads to a speedup of up to 2x for [memory-bound](https://modal.com/gpu-glossary/perf/memory-bound)
# workloads like the decode phase of LLM inference.
# It also doubles the [arithmetic bandwidth](https://modal.com/gpu-glossary/perf/arithmetic-bandwidth)
# for [compute-bound](https://modal.com/gpu-glossary/perf/compute-bound)
# workloads like the prefill phase of LLM inference.

N_GPUS = 2
GPU = f"H100!:{N_GPUS}"

# Actual speedups are generally less than what you get from "napkin math" based on available bandwidths --
# we observed a speedup of about 30% moving from one to two H100s when developing this example.
# We recommend [application-specific benchmarking](https://modal.com/llm-almanac/how-to-benchmark)
# guided by [published generic benchmarks](https://modal.com/llm-almanac/advisor).

# ### Loading and cacheing the model weights

# We'll serve [Alibaba's Qwen 3 LLM](https://www.alibabacloud.com/blog/alibaba-introduces-qwen3-setting-new-benchmark-in-open-source-ai-with-hybrid-reasoning_602192).
# For lower latency, we pick a smaller model (8B params)
# in a lower precision floating point format (FP8).
# This reduces the amount of data that needs to be loaded
# [from GPU RAM into SM SRAM](https://modal.com/gpu-glossary/perf/memory-bandwidth)
# in each forward pass.

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
MODEL_REVISION = (
    "220b46e3b2180893580a4454f21f22d3ebb187d3"  # latest commit as of 2026-01-01
)

# We load the model [from the Hugging Face Hub](https://huggingface.co/collections/Qwen/qwen3),
# so we'll need their Python package.

sglang_image = sglang_image.uv_pip_install("huggingface-hub==0.36.0")

# We don't want to load the model from the Hub every time we start the server.
# We can load it much faster from a [Modal Volume](https://modal.com/docs/guide/volumes).
# Typical speeds are around one to two GB/s.

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
MODEL_PATH = f"{HF_CACHE_PATH}/{MODEL_NAME}"

# In addition to pointing the Hugging Face Hub at the path
# where we mount the Volume, we also
# [turn on "high performance" downloads](https://huggingface.co/docs/hub/en/models-downloading#faster-downloads),
# which can fully saturate our network bandwidth.

sglang_image = sglang_image.env(
    {"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"}
)

# ### Cacheing compilation artifacts

# Model weights aren't the only thing we want to cache.

# As a rule, LLM inference servers like SGLang don't directly provide their own kernels.
# They draw high-performance kernels from a variety of sources.

# As of version `0.5.6`, SGLang's default kernel backend
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


# We run this Python function on Modal as part of building the Image
# so that it has access to the appropriate GPU and the caches for our model and compilaton artifacts.

sglang_image = sglang_image.run_function(
    compile_deep_gemm,
    volumes={DG_CACHE_PATH: DG_CACHE_VOL, HF_CACHE_PATH: HF_CACHE_VOL},
    gpu=GPU,
)


# ## Speed up cold starts with GPU snapshotting

# Modal is a serverless compute platform, so all of your
# inference services automatically scale up and down to handle
# variable load.

# Scaling up a new replica requires quite a bit of work --
# loading up Python and system packages, loading model weights,
# setting up the inference engine, and so on.

# We can skip over and speed up a bunch of this work
# when spinning up new replicas after the first
# by directly booting from a [memory snapshot](https://modal.com/docs/guide/memory-snapshot),
# which contains the exact in-memory representation of our server just before it begins taking requests.

# Most applications can be snapshot and experience substantial speedups (2x to 10x,
# see [our initial benchmarks here](https://modal.com/blog/gpu-mem-snapshots)).
# However, it generally requires some extra work to adapt the application code.

# For instance, we here set an environment variable that improves the compatibility of
# the [Torch Inductor compiler](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
# with GPU snapshotting and enables

sglang_image = sglang_image.env(
    {"TORCHINDUCTOR_COMPILE_THREADS": "1", "TMS_INIT_ENABLE_CPU_BACKUP": "1"}
)

# Below, we walk through a few steps required to make an SGLang server compatible with snapshots.

# ### Sleeping and waking an SGLang server

# We prepare our SGLang inference server for snapshotting by first sending
# a few requests to "warm it up", ensuring that it is fully ready to process requests.
# Then we "put it to sleep", moving non-essential data out of GPU memory,
# with a request to `/release_memory_occupation`.
# At this point, we can take a memory snapshot.
# Upon snapshot restoration, we "wake up" the server
# with a request to `/resume_memory_occupation`.

# We use the [`requests` library](https://requests.readthedocs.io/en/latest/)
# to send ourselves these HTTP requests on
# [`localhost`/`127.0.0.1`](https://superuser.com/questions/31824/why-is-localhost-ip-127-0-0-1).

with sglang_image.imports():
    import requests


def warmup():
    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=10
        ).raise_for_status()


def sleep():
    requests.post(
        f"http://127.0.0.1:{PORT}/release_memory_occupation", json={}
    ).raise_for_status()


def wake_up():
    requests.post(
        f"http://127.0.0.1:{PORT}/resume_memory_occupation", json={}
    ).raise_for_status()


# ## Define the inference server and infrastructure

# ### Selecting infrastructure to minimize latency

# Minimizing latency requires geographic co-location of clients and servers.

# So for low latency LLM inference services on Modal, you must select a
# [cloud region](https://modal.com/docs/guide/region-selection)
# for both the GPU-accelerated containers running inference
# and for the internal Modal proxies that forward requests to them
# as part of defining a `modal.experimental.http_server`.

# Here, we assume users are mostly in the northern half of the Americas
# and select a single cloud region, `us-east`, to serve them.

REGION = "us-east"

# This should result in at most a few dozen milliseconds of round-trip time.

# For production-scale LLM inference services, there are generally
# enough requests to justify keeping at least one replica running at all times.
# Having a "warm" or "live" replica reduces latency by skipping slow initialization work
# that occurs when new replica boot up (a ["cold start"](https://modal.com/docs/guide/cold-start)).
# For LLM inference servers, that latency runs into the seconds or few tens of seconds,
# even with snapshotting.

# To ensure at least one container is always available,
# we can set the `min_containers` of our Modal Function
# to `1` or more.

# However, since this is sample code, we'll set it to `0`.

MIN_CONTAINERS = 0  # set to 1 to ensure one replica is always ready

# Finally, we need to decide how we will scale up and down replicas
# in response to load. Without autoscaling, users' requests will queue
# when the server becomes overloaded. Even apart from queueing, responses
# generally become slower per user above a certain minimum number of
# concurrent requests.

# So we set a target for the number of inputs to run on a single container
# with [`modal.concurrent`](https://modal.com/docs/reference/modal.concurrent).
# For details, see [the guide](https://modal.com/docs/guide/concurrent-inputs).

TARGET_INPUTS = 20

# Generally, this choice needs to be made as part of
# [LLM inference engine benchmarking](https://modal.com/llm-almanac/how-to-benchmark).

# ### Controlling container lifecycles with `Modal.Cls`

# We wrap up all of the choices we made about the infrastructure
# of our inference server into a number of Python decorators
# that we apply to a Python class that encapsulates the logic
# to run our server.

# The key decorators are:

# - [`@app.cls`](https://modal.com/docs/guide/lifecycle-functions) to define the core of our service.
# We attach our Image, request a GPU, attach our cache Volumes, specify the region, and configure auto-scaling.
# See [the reference documentation](https://modal.com/docs/reference/modal.App#cls) for details.

# - `@modal.experimental.http_server` to turn our Python code into an HTTP server
# (i.e. fronting all of our containers with a proxy with a URL). The wrapped code
# needs to eventually listen for HTTP connections on the provided `port`.

# - [`@modal.concurrent`](https://modal.com/docs/guide/concurrent-inputs) to specify how many
# requests our server can handle before we need to scale up.

# - [`@modal.enter` and `@modal.exit`](https://modal.com/docs/guide/lifecycle-functions) to indicate
# which methods of the class should be run when starting the server and shutting it down. The `enter`
# methods also define what code is run before memory snapshot creation (`snap=True`) and after memory snapshot restoration (`snap=False`).

# Modal considers a new replica ready to receive inputs once the `modal.enter` methods have exited
# and the container accepts connections.
# To ensure that we actually finish setting up our server before we are marked ready for inputs,
# we define a helper function to check whether the server is finished setting up.


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


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


# With all this in place, we are ready to define our high-performance, low-latency
# LLM inference server.

app = modal.App(name="example-sglang-low-latency")
PORT = 8000


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
    port=PORT,  # wrapped code must listen on this port
    proxy_regions=[REGION],  # location of proxies, should be same as Cls region
    exit_grace_period=5,  # seconds, time to finish up requests when closing down
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class SGLang:
    @modal.enter(snap=True)
    def startup(self):
        """Start the SGLang server and block until it is healthy, then warm it up and put it to sleep."""

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
            "--tp",  # use all GPUs to split up tensor-parallel operations
            f"{N_GPUS}",
            "--cuda-graph-max-bs",  # only capture CUDA graphs for batch sizes we're likely to observe
            f"{TARGET_INPUTS * 2}",
            "--enable-metrics",  # expose metrics endpoints for telemetry
            "--enable-memory-saver",  # enable offload, for snapshotting
            "--enable-weights-cpu-backup",  # enable offload, for snapshotting
        ]

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()  # for snapshotting
        sleep()

    @modal.enter(snap=False)
    def wake_up(self):
        wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()


## Deploy the server

# To deploy the server on Modal, just run

# ```bash
# modal deploy sglang_low_latency.py
# ```

# This will create a new App on Modal and build the container image for it if it hasn't been built yet.

## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-sglang-low-latency-sglang.us-east.modal.direct`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-sglang-low-latency-sglang.us-east.modal.direct/docs`.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands.
# For simple routes, you can even send a request directly from the docs page.

# Note: when no replicas are available, Modal will respond with
# the [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503).
# In your browser, you can just hit refresh until the docs page appears.
# You can see the status of the applicaton and its containers on your [Modal dashboard](https://modal.com/apps).

## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that hits the server with a simple client.

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
# is starting up -- timeouts on the client and 5XX responses from the server.
# Modal returns the [503 Service Unavailable status](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Status/503)
# when an `experimental.http_server` has no live replicas.


async def probe(url, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [{"role": "user", "content": "Tell me a joke."}]

    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() < deadline:
            try:
                await _send_request(session, "llm", messages)
                return
            except asyncio.TimeoutError:
                await asyncio.sleep(1)
            except aiohttp.client_exceptions.ClientResponseError as e:
                if e.status == 503:
                    await asyncio.sleep(1)
                    continue
                raise e
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

# Using `modal run` creates an ephemeral Modal App, rather than a deployed Modal App.
# Ephemeral Modal Apps are short-lived, so they turn off memory snapshotting.

# To test the memory snapshot version of the server,
# first deploy it with `modal deploy`
# and then hit it with a client.

# You should observe startup improvements
# after a handful of cold starts
# (usually less than five).
# If you want to see the speedup during a test,
# we recommend heading to the deployed App in your
# [Modal dashboard](https://modal.com/apps)
# and manually stopping containers after they have served a request
# to ensure turnover.

# You can use the client code below to test the endpoint.
# It can be run with the command

# ```bash
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
