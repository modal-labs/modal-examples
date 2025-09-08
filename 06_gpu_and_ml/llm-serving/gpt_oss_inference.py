# ---
# pytest: false
# ---

# # Run OpenAI's gpt-oss model with vLLM

# ## Background

# [gpt-oss](https://openai.com/index/introducing-gpt-oss/) is a reasoning model
# that comes in two flavors: `gpt-oss-120B` and `gpt-oss-20B`. They are both Mixture
# of Experts (MoE) models with a low number of active parameters, ensuring they
# combine good world knowledge and capabilities with fast inference.

# We describe a few of its notable features below.

# ### MXFP4

# OpenAI's gpt-oss models use a fairly uncommon 4bit [`mxfp4`](https://arxiv.org/abs/2310.10537) floating point
# format for the MoE layers. This "block" quantization format combines `e2m1` floating point numbers
# with blockwise scaling factors. The attention operations are not quantized.

# ### Attention Sinks

# Attention sink models allow for longer context lengths without sacrificing output quality. The vLLM team
# added [attention sink support](https://huggingface.co/kernels-community/vllm-flash-attn3)
# for Flash Attention 3 (FA3) in preparation for this release.

# ### Response Format

# GPT-OSS is trained with the [harmony response format](https://github.com/openai/harmony) which enables models
# to output to multiple channels for chain-of-thought (CoT) and input tool-calling preambles along with regular text responses.
# We'll stick to a simpler format here, but see [this cookbook](https://cookbook.openai.com/articles/openai-harmony)
# for details on the new format.

# ## Set up the container image

# We'll start by defining a [custom container `Image`](https://modal.com/docs/guide/custom-container) that
# installs all the necessary dependencies to run vLLM and the model. This includes a special-purpose vLLM prerelease
# and a nightly PyTorch install for Triton support.

import json
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.10.1+gptoss",
        "huggingface_hub[hf_transfer]==0.34",
        pre=True,
        extra_options="--extra-index-url https://wheels.vllm.ai/gpt-oss/ --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match",
    )
)


# ## Download the model weights

# We'll be downloading OpenAI's model from Hugging Face. We're running
# the 20B parameter model by default but you can easily switch to [the 120B model](https://huggingface.co/openai/gpt-oss-120b),
# which also fits in a single H100 or H200 GPU.

MODEL_NAME = "openai/gpt-oss-20b"
MODEL_REVISION = "d666cf3b67006cf8227666739edf25164aaffdeb"

# Although vLLM will download weights from Hugging Face on-demand, we want to
# cache them so we don't do it every time our server starts. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes)
# for our cache. Modal Volumes are essentially a "shared disk" that all Modal
# Functions can access like it's a regular disk. For more on storing model
# weights on Modal, see [this guide](https://modal.com/docs/guide/model-weights).

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# The first time you run a new model or configuration with vLLM on a fresh machine,
# a number of artifacts are created. We also cache these artifacts.

vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# There are a number of compilation settings for vLLM. Compilation improves inference performance
# but incur extra latency at engine start time. We offer a high-level variable for controlling this trade-off.

FAST_BOOT = False  # slower boots but faster inference

# Among the artifacts that are created at startup are CUDA graphs,
# which allow the replay of several kernel launches for the price of one,
# reducing CPU overhead. We over-ride the defaults with a smaller number of sizes
# that we think better balances latency from future JIT CUDA graph generation
# and startup latency.

MAX_INPUTS = 32  # how many requests can one replica handle? tune carefully!
CUDA_GRAPH_CAPTURE_SIZES = [  # 1, 2, 4, ... MAX_INPUTS
    1 << i for i in range((MAX_INPUTS).bit_length())
]

# ## Build a vLLM engine and serve it

# The function below spawns a vLLM instance listening at port 8000, serving requests to our model.

app = modal.App("example-gpt-oss-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=30 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
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
    ]

    # enforce-eager disables both Torch compilation and CUDA graph capture
    # default is no-enforce-eager. see the --compilation-config flag for tighter control
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    if not FAST_BOOT:  # CUDA graph capture is only used with `--enforce-eager`
        cmd += [
            "-O.cudagraph_capture_sizes="
            + str(CUDA_GRAPH_CAPTURE_SIZES).replace(" ", "")
        ]

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


# ## Deploy the server

# To deploy the API on Modal, just run

# ```bash
# modal deploy gpt_oss_inference.py
# ```

# This will create a new app on Modal, build the container image for it if it hasn't been built yet,
# and deploy the app.

# ## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that does a healthcheck and then hits the server.

# If you execute the command

# ```bash
# modal run gpt_oss_inference.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# We set up the system prompt with low reasoning effort to run
# inference a bit faster. For the best ergonomics we recommend using
# the [harmony API](https://cookbook.openai.com/articles/openai-harmony#example-system-message),
# which can be installed with `pip install openai-harmony`.


@app.local_entrypoint()
async def test(test_timeout=30 * MINUTES, user_content=None, twice=True):
    url = serve.get_web_url()
    system_prompt = {
        "role": "system",
        "content": f"""You are ChatModal, a large language model trained by Modal.
        Knowledge cutoff: 2024-06
        Current date: {datetime.now(timezone.utc).date()}
        Reasoning: low
        \\# Valid channels: analysis, commentary, final. Channel must be included for every message.
        Calls to these tools must go to the commentary channel: 'functions'.""",
    }

    if user_content is None:
        user_content = "Explain what the Singular Value Decomposition is."

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": user_content},
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
            messages[0]["content"] += "\nTalk like a pirate, matey."
            print(f"Re-sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {"messages": messages, "model": model, "stream": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    t = time.perf_counter()
    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=10 * MINUTES
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

            if "content" in delta:
                print(delta["content"], end="")  # print the content as it comes in
            elif "reasoning_content" in delta:
                print(delta["reasoning_content"], end="")
            else:
                raise ValueError(f"Unsupported response delta: {delta}")
    print("")
    print(f"Time to Last Token: {time.perf_counter() - t:.2f} seconds")
