# ---
# deploy: true
# cmd: ["python", "06_gpu_and_ml/llm-serving/ministral3_inference.py"]
# ---

# # Serverless Ministral 3 with vLLM and Modal

# In this example, we show how to serve Mistral's Ministral 3 vision-language models on Modal.

# The [Ministral 3](https://huggingface.co/collections/mistralai/ministral-3-more) model series
# performs competitively with the Qwen 3-VL model series on benchmarks
# (see model cards for details).

# We also include instructions for cutting cold start times
# for long-running deployments by an order of magnitude using Modal's
# [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# We'll use the [vLLM inference server](https://docs.vllm.ai).
# vLLM can be installed with `uv pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

import json
import socket
import subprocess
from typing import Any

import aiohttp
import modal

MINUTES = 60  # seconds
VLLM_PORT = 8000

app = modal.App("example-ministral3-inference")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm~=0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
)

# ## Download the Ministral weights

# We also need to download the model weights.
# We'll retrieve them from the Hugging Face Hub.

# To speed up the model load, we'll toggle the `HIGH_PERFORMANCE`
# flag for Hugging Face's [Xet backend](https://huggingface.co/docs/hub/en/xet/index).

vllm_image = vllm_image.env({"HF_XET_HIGH_PERFORMANCE": "1"})

# The [Ministral 3 model series](https://huggingface.co/collections/mistralai/ministral-3-more)
# contains a variety of models:

# - 3B, 8B, and 14B sizes
# - base models and instruction & reasoning fine-tuned models
# - BF16 and FP8 quantizations

# All are available under the Apache 2.0 open source license.

# We'll use the FP8 instruct variant of the 8B model:

MODEL_NAME = "mistralai/Ministral-3-8B-Instruct-2512"

# Native hardware support for FP8 formats in [Tensor Cores](https://modal.com/gpu-glossary/device-hardware/tensor-core)
# is limited to the latest [Streaming Multiprocessor architectures](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
# like those of Modal's [Hopper H100/H200 and Blackwell B200 GPUs](https://modal.com/blog/announcing-h200-b200).

# At 80 GB VRAM, a single H100 GPU has enough space to store the 8B FP8 model weights (~8 GB)
# and a very large KV cache. A single H100 is also enough to serve the 14B model in full precision,
# but without as much room for KV (though still enough to handle the full sequence length).

N_GPU = 1

# ### Cache with Modal Volumes

# Modal Functions are serverless: when they aren't being used,
# their underlying containers spin down and all ephemeral resources,
# like GPUs, memory, network connections, and local disks are released.

# We can preserve saved files by mounting a
# [Modal Volume](https://modal.com/docs/guide/volumes) --
# a persistent, remote filesystem.

# We'll use two Volumes: one for weights from Hugging Face
# and one for compilation artifacts from vLLM.

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ## Serve Ministral 3 with vLLM

# We serve Ministral 3 on Modal by spinning up a Modal Function
# that acts as a [`web_server`](https://modal.com/docs/guide/webhooks)
# and spins up a vLLM server in a subprocess
# (via the `vllm serve` command).

# ### Improve cold start time with snapshots

# Starting up a vLLM server can be slow --
# tens of seconds to minutes. Much of that time
# is spent on JIT compilation of inference code.

# We can skip most of that work and reduce startup times by a factor of 10
# using Modal's [memory snapshots](https://modal.com/docs/guide/memory-snapshot),
# which serialize the contents of CPU and GPU memory.

# This adds quite some complexity to the code.
# If you're looking for a minimal example, see
# our [`vllm_inference` example here](https://modal.com/docs/examples/vllm_inference).

# We'll need to set a few extra configuration values:

vllm_image = vllm_image.env(
    {
        "VLLM_SERVER_DEV_MODE": "1",  # allow use of "Sleep Mode"
        "TORCHINDUCTOR_COMPILE_THREADS": "1",  # improve compatibility with snapshots
    }
)

# Setting the `DEV_MODE` flag allows us to use the `sleep`/`wake_up` endpoints
# to toggle the server in and out of "sleep mode".

with vllm_image.imports():
    import requests


def sleep(level=1):
    requests.post(
        f"http://localhost:{VLLM_PORT}/sleep?level={level}"
    ).raise_for_status()


def wake_up():
    requests.post(f"http://localhost:{VLLM_PORT}/wake_up").raise_for_status()


# Sleep Mode helps with memory snapshotting.
# When the server is asleep, model weights are offloaded to CPU memory and the KV cache is emptied.
# For details, see the [vLLM docs](https://docs.vllm.ai/en/stable/features/sleep_mode/).

# We'll also need two helper functions.
# Ther first, `wait_ready`, busy-polls the server until it is live.


def wait_ready(proc: subprocess.Popen):
    while True:
        try:
            socket.create_connection(("localhost", VLLM_PORT), timeout=1).close()
            return
        except OSError:
            if proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {proc.returncode}")


# Once the server is live, we `warmup` inference with a few requests.
# This is important for capturing non-serializable JIT compilation artifacts,
# like CUDA graphs and some Torch compilation outputs,
# in our snapshot.


def warmup():
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Who are you?"}],
        "max_tokens": 16,
    }

    for ii in range(3):
        requests.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=300,
        ).raise_for_status()


# ### Define the server

# We construct our web-serving Modal Function
# by decorating a regular Python class.
# The decorators include a number of configuration
# options for deployment, including resources like GPUs and Volumes
# and timeouts on container scaledown.
# You can read more about the options
# [here](https://modal.com/docs/reference/modal.App#function).

# We control memory snapshotting and container start behavior
# by decorating the methods of the class.

# We start the server, warm it up, and then put it to sleep
# in the `start` method. This method has the `modal.enter`
# decorator to ensure it runs when a new container starts
# and we pass `snap=True` to turn on memory snapshotting.

# The following method, `wake_up`, calls the `wake_up`
# endpoint and then waits for the server to be ready.
# It is run after the `start` method because it is defined later
# in the code and also has the `modal.enter` decorator.
# It has `snap=False` so that it isn't included in the snapshot.

# Finally, we connect the vLLM server to the Internet
# using the [`modal.web_server`](https://modal.com/docs/guide/webhooks#non-asgi-web-servers) decorator.


@app.cls(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
class VllmServer:
    @modal.enter(snap=True)
    def start(self):
        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--gpu_memory_utilization",
            str(0.95),
        ]

        # assume multiple GPUs are for splitting up large matrix multiplications
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        # add mistral config arguments
        cmd += [
            "--tokenizer_mode",
            "mistral",
            "--config_format",
            "mistral",
            "--load_format",
            "mistral",
            "--tool-call-parser",
            "mistral",
            "--enable-auto-tool-choice",
        ]

        # add config arguments for snapshotting

        cmd += [
            "--enable-sleep-mode",
            # make KV cache predictable / small
            "--max-num-seqs",
            "2",
            "--max-model-len",
            "12288",
            "--max-num-batched-tokens",
            "12288",
        ]

        print(*cmd)

        self.vllm_proc = subprocess.Popen(cmd)

        wait_ready(self.vllm_proc)

        warmup()

        sleep()

    @modal.enter(snap=False)
    def wake_up(self):
        wake_up()
        wait_ready(self.vllm_proc)

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.vllm_proc.terminate()


# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy ministral3_inference.py
# ```

# This will create a new app on Modal, build the container image for it if it hasn't been built yet,
# and deploy the app.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-ministral3-inference-serve.modal.run`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-ministral-inference-serve.modal.run/docs`.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands.

# For simple routes like `/health`, which checks whether the server is responding,
# you can even send a request directly from the docs.

# To interact with the API programmatically in Python, we recommend the `openai` library.

# ## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that does a healthcheck and then hits the server.

# If you execute the command

# ```bash
# modal run ministral3_inference.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None, twice=True):
    url = VllmServer().serve.get_web_url()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

        content = [
            {
                "type": "text",
                "text": "What action do you think I should take in this situation?"
                " List all the possible actions and explain why you think they are good or bad.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200
        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages, timeout=1 * MINUTES)
        if twice:
            messages[0]["content"] = """Yousa culled Jar Jar Binks.
            Always be talkin' in da Gungan style, like thisa, okeyday?
            Helpin' da user with big big enthusiasm, makin' tings bombad clear!"""
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages, timeout=1 * MINUTES)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list, timeout: int
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        "temperature": 0.15,
    }

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=timeout
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
            print(chunk["choices"][0]["delta"]["content"], end="")
    print()


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
# python ministral3_inference.py
# ```

if __name__ == "__main__":
    import asyncio

    # after deployment, we can use the class from anywhere
    VllmServer = modal.Cls.from_name("example-ministral3-inference", "VllmServer")
    server = VllmServer()

    async def test(url):
        messages = [{"role": "user", "content": "Tell me a joke."}]
        async with aiohttp.ClientSession(base_url=url) as session:
            await _send_request(session, "llm", messages, timeout=10 * MINUTES)

    try:
        print("calling inference server")
        asyncio.run(test(server.serve.get_web_url()))
    except modal.exception.NotFoundError:
        raise Exception(
            f"To take advantage of GPU snapshots, deploy first with modal deploy {__file__}"
        )
