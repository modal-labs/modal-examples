# # Serve Ministral 3 with vLLM

# In this example, we show how to serve Mistral's Ministral 3 vision-language models on Modal.

# The [Ministral 3](https://huggingface.co/collections/mistralai/ministral-3-more) model series
# performs competitively with the Qwen 3-VL model series on benchmarks
# (see model cards for details).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# We'll use the [vLLM inference server](https://docs.vllm.ai).
# vLLM can be installed with `uv pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

import json
from typing import Any

import aiohttp
import modal

app = modal.App("example-ministral3-inference")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
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

# At 80 GB VRAM, a single H100 GPU has enough space to store the 3B FP8 model weights (~3 GB)
# and a very large KV cache. A single H100 is also enough to serve the 14B model in full precision,
# but without as much room for KV (though still enough to handle the full sequence length).

N_GPU = 1

# ### Cacheing with Modal Volumes

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

# ## Serving Ministral 3 with vLLM

# We serve Ministral 3 on Modal by spinning up a Modal Function
# that acts as a [`web_server`](https://modal.com/docs/guide/webhooks)
# and spins up a vLLM server in a subprocess
# (via the `vllm serve` command).

# The majority of the code in our Python function
# constructs the arguments to this command
# to configure the vLLM server.

# For autoscaling vLLM deployments,
# one of the key knobs to turn is the amount of
# work done at server startup -- typically
# balancing [cold start performance](https://modal.com/docs/guide/cold-start)
# and performance per request.

# We opt for faster boots.
# We add a global variable to abstract away
# the details of this choice.

FAST_BOOT = True

# We construct our web-serving Modal Function
# by decorating a regular Python function.
# The decorators include a number of configuration
# options for deployment, including resources like GPUs and Volumes
# and timeouts on container scaledown.
# You can read more about the options
# [here](https://modal.com/docs/reference/modal.App#function).

MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

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
        str(0.99),
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

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

    print(*cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy vllm_inference.py
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

# ## Testing the server

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
    url = serve.get_web_url()

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
        await _send_request(session, "llm", messages)
        if twice:
            messages[0]["content"] = """Yousa culled Jar Jar Binks.
            Always be talkin' in da Gungan style, like thisa, okeyday?" +
            Helpin' da user with big big enthusiasm, makin' tings bombad clear!"""
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
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
            print(chunk["choices"][0]["delta"]["content"], end="")
    print()
