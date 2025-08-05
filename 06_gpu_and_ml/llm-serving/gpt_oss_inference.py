# ---
# pytest: false
# ---

# # Run OpenAI's first OSS model with vLLM

# It's hard to remember now but OpenAI was founded with the mission of ensuring
# that the most powerful AI models wouldn't be controlled exclusively by
# for-profit corporations. But the compute-demands of AI couldn't be ignored and
# they partnered with Microsoft and surprised the world with ChatGPT in December
# 2022, a bot that made us all question if the Turing test was on the verge of
# being solved—hitting 100 million users in just two months. But the open source
# community pushed back hard, people wanted modifiable models they could actually
# control, and LLaMA, Qwen, and DeepSeek all became serious rivals that trailed
# closed-source performance by only 6-12 months year after year. Now OpenAI is
# teetering back to its roots, releasing their very first open source model.

# The cracked engineers at vLLM have brought 0-day support for it and we're supplying
# the infrastructure so anyone can run it.

# ## Set up the container image

# We'll start by defining a [custom container `Image`](https://modal.com/docs/guide/custom-container) that
# install all the necessary dependencies to run vLLM and the model. This will include `amd-quark` which is
# AMD's quantization library that support [`mxfp4`](https://en.wikipedia.org/wiki/Block_floating_point#Microscaling_(MX)_Formats)
# quantization. OpenAI has provided the model is `mxfp4` format.

import json
from typing import Any

import aiohttp
import modal
import time

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
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1", # faster model transfers
        "VLLM_USER_V1": "1", # latest engine
    })  
)


# ## Download the model weights

# We'll be downloading OpenAI's model straight from huggingface. We're running the 20B parameter 
# model by default but you can easily switch to the 20B model too.
MODEL_NAME = "openai/gpt-oss-20b"

# Although vLLM will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use [Modal Volumes](https://modal.com/docs/guide/volumes) for our cache.
# Modal Volumes are essentially a "shared disk" that all Modal Functions can access like it's a regular disk. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).

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
    timeout=10 * MINUTES,  # how long should we wait for container start?
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
async def test(test_timeout=10 * MINUTES, content=None, twice=True):
    url = serve.get_web_url()

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

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    t = time.perf_counter()
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
    print(f"TTLT_s: {time.perf_counter() - t:.2f} seconds")
    print()


