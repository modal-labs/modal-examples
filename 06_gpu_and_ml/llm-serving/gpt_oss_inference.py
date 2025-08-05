# ---
# pytest: false
# ---

# # Run OpenAI's first OSS model with vLLM

# Run OpenAI's first open source model with vLLM.

# ## Background
# ### Overview
# GPT-OSS is a reasoning model that comes in two flavors gpt-oss-120B and gpt-oss-20B. They are both
# Mixture of Experts (MoE) models that allow for a low number of active parameters, 5.1B and 3.6B respectively.

# ### MXFP4
# OpenAI's GPT-OSS models use [`mxfp4`](https://arxiv.org/abs/2310.10537) precision for the MoE layers
# during training, this is a block floating point format that allow for more efficient training and inference.

# ### Attention Sinks
# Attention sink models allow for longer context lengths without sacrificing output quality. The vLLM team
# added [attention sink support](https://huggingface.co/kernels-community/vllm-flash-attn3)
# for Flash Attention 3 (FA3) in prep for this release.

# ### Response Format
# GPT-OSS is trained with the [harmony response format](https://github.com/openai/harmony) which enables models
# to output to multipe channels for chain-of-thought (CoT), and input tool calling preambles along with regular responses.

# ## Set up the container image

# We'll start by defining a [custom container `Image`](https://modal.com/docs/guide/custom-container) that
# installs all the necessary dependencies to run vLLM and the model. This includes a custom vllm version
# and a nightly pytorch install so that we can run gpt-oss running.

import json
import time
from typing import Any

import aiohttp
import modal

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_pip_install(
        "vllm==0.10.1+gptoss",
        pre=True,
        extra_options="--extra-index-url https://wheels.vllm.ai/gpt-oss/ --extra-index-url https://download.pytorch.org/whl/nightly/cu128 --index-strategy unsafe-best-match",
    )
    .uv_pip_install(
        "huggingface_hub[hf_transfer]==0.34",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model transfers
            "VLLM_USER_V1": "1",  # latest engine
        }
    )
)


# ## Download the model weights

# We'll be downloading OpenAI's model from Hugging Face. We're running
# the 20B parameter model by default but you can easily switch to [the 120B model](https://huggingface.co/openai/gpt-oss-120b).
MODEL_NAME = "openai/gpt-oss-20b"

# Although vLLM will download weights from Hugging Face on-demand, we want to
# cache them so we don't do it every time our server starts. We'll use [Modal Volumes](https://modal.com/docs/guide/volumes)
#  for our cache. Modal Volumes are essentially a "shared disk" that all Modal
# Functions can access like it's a regular disk. For more on storing model
# weights on Modal, see [this guide](https://modal.com/docs/guide/model-weights).

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("example-gpt-oss-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000
FAST_BOOT = True


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
        # "--revision",
        # MODEL_REVISION,
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

    # assume multiple GPUs are for splitting up large matrix multiplications
    cmd += ["--tensor-parallel-size", str(N_GPU)]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def test(test_timeout=30 * MINUTES, user_content=None):
    url = serve.get_web_url()
    system_prompt_content: str = """
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-05
Reasoning: low
# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.
"""
    system_prompt = {
        "role": "system",
        "content": system_prompt_content,
    }

    if user_content is None:
        user_content = "Explain what MXFP4 quantization is."

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
