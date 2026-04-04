# ---
# pytest: false
# ---

# # Run OpenAI-compatible LLM inference with Gemma and vLLM

# In this example, we show how to run a vLLM server in OpenAI-compatible mode on Modal.

# LLMs do more than just model language: they chat, they produce JSON and XML, they run code, and more.
# This has complicated their interface far beyond "text-in, text-out".
# OpenAI's API has emerged as a standard for that interface,
# and it is supported by open source LLM serving frameworks like [vLLM](https://docs.vllm.ai/en/latest/).

# This example is intended to demonstrate the basics of deploying LLM inference on Modal.
# For more on how to optimize performance, see
# [this guide](https://modal.com/docs/guide/high-performance-llm-inference)
# and check out our
# [LLM Engineer's Almanac](https://modal.com/llm-almanac).

# Our examples repository also includes scripts for running clients and load-testing for OpenAI-compatible APIs
# [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# vLLM can be installed with `uv pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

import json
from typing import Any

import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.19.0",
    )
    .uv_pip_install(  # as of vllm 0.19.0, must install transformers separately to use Gemma 4
        "transformers==5.5.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

# ## Download the model weights

# We'll be running a pretrained foundation model --
# [Goole's Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/).
# It can also take images, video, and audio as inputs,
# though we won't use that here.

# We'll use the 26BA4B variant, [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).
# This variant is trained with reasoning capabilities, which allow it to
# enhance the quality of its generated responses.
# It has `26B`illion parameters, of which `4B`illion are `A`ctive
# in processing of each token.

# You can swap this model out for another by changing the strings below,
# though you might also need to adjust some of the server configuration as well.
# A single H200 GPU has enough VRAM to store this 26,000,000,000 parameter model
# along with a large KV cache.


MODEL_NAME = "google/gemma-4-26B-A4B-it"
MODEL_REVISION = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"  # avoid nasty surprises when repos update!

# Although vLLM will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use [Modal Volumes](https://modal.com/docs/guide/volumes) for our cache.
# Modal Volumes are essentially a "shared disk" that all Modal Functions can access like it's a regular disk.
# For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# We'll also cache some of vLLM's JIT compilation artifacts in a Modal Volume.

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ## Configuring vLLM

# ### Trading off fast boots and token generation performance

# vLLM has embraced dynamic and just-in-time compilation to eke out additional performance without having to write too many custom kernels,
# e.g. via the Torch compiler and CUDA graph capture.
# These compilation features incur latency in exchange for lowered latency and higher throughput during generation.
# This latency is typically tens of seconds to a few minutes, reduced to about ten seconds when loaded from the cache.
# We make this trade-off controllable with the `FAST_BOOT` variable below.

FAST_BOOT = False

# If you're running an LLM service that frequently scales from 0 (frequent ["cold starts"](https://modal.com/docs/guide/cold-start))
# you might want to set this to `True`, or consider [GPU memory snapshots](https://modal.com/docs/guide/memory-snapshots).
# It's also useful to set this when you're iterating on the server configuration.

# If you're running an LLM service that usually has multiple replicas running, then set this to `False` for improved performance.

# See the code below for details on the parameters that `FAST_BOOT` controls.

# For more on the performance you can expect when serving your own LLMs, see
# [our LLM engine performance benchmarks](https://modal.com/llm-almanac).

# ## Build a vLLM engine and serve it

# The function below spawns a vLLM instance listening at port 8000, serving requests to our model.
# We wrap it in the [`@modal.web_server` decorator](https://modal.com/docs/guide/webhooks#non-asgi-web-servers)
# to connect it to the Internet.

# The server runs in an independent process, via `subprocess.Popen`, and only starts accepting requests
# once the model is spun up and the `serve` function returns.


app = modal.App("example-vllm-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=100,
)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import json
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--async-scheduling",
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    # add model-specific configuration
    cmd += [
        # skip multimedia support, just language
        "--limit-mm-per-prompt",
        f"'{json.dumps({'image': 0, 'video': 0, 'audio': 0})}'",
        # enable reasoning and tool use
        "--enable-auto-tool-choice",
        "--reasoning-parser gemma4",
        "--tool-call-parser gemma4",
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
# something like `https://your-workspace-name--example-vllm-inference-serve.modal.run`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-vllm-inference-serve.modal.run/docs`.
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
# modal run vllm_inference.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None, twice=True):
    url = await serve.get_web_url.aio()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

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
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}
    # explicitly enable thinking for this model
    payload["chat_template_kwargs"] = {"enable_thinking": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers
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
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
            if content:
                print(content, end="")
            else:
                print("\n", chunk)
    print()


# We also include a basic example of a load-testing setup using
# `locust` in the `load_test.py` script [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible):

# ```bash
# modal run openai_compatible/load_test.py
# ```
