# ---
# deploy: true
# ---

# # Serverless Tokasaurus (Qwen2-7B-Instruct)

# In this example, we demonstrate how to use the Tokasaurus framework to serve Qwen2-7B-Instruct model
# at very high throughput.

# ## Overview

# This guide is intended to document two things:
# the general process for building Tokasaurus on Modal
# and a specific configuration for serving the Qwen2-7B-Instruct model.

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# Tokasaurus can be installed with `pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

# To take advantage of optimized kernels for CUDA 12.8, we install PyTorch, and their dependencies
# via an `extra` Python package index.

import json

import aiohttp
import modal

toka_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.1-devel-ubuntu22.04", add_python="3.12"
    )  # since tokasaurus==0.0.2 uses torch==2.6.0
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode tokasaurus==0.0.2 huggingface_hub[hf_transfer]==0.33.0"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# ## Download the model weights

# We'll be running a fine-tuned instruction-following model -- Qwen2-7B-Instruct
# that's trained to chat and follow instructions.

MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
MODEL_REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"  # avoid nasty surprises when repos update!

# Although Tokasaurus will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use [Modal Volumes](https://modal.com/docs/guide/volumes) for our cache.
# Modal Volumes are essentially a "shared disk" that all Modal Functions can access like it's a regular disk.

app_name = "example-tokasaurus-throughput"

hf_cache_vol = modal.Volume.from_name(f"{app_name}-hf-cache", create_if_missing=True)

# ## Tuning the engine

# You can tune the engine for high throughput by adjusting the following two parameters:
# - `max_tokens_per_forward`: max tokens processed per forward pass
# - `max_seqs_per_forward`: max sequences processed per forward pass

# These two parameters work together to [maximize batch size and throughput](https://github.com/ScalingIntelligence/tokasaurus#managing-gpu-memory-with-kv-cache-limits-and-concurrency-controls).
# Increase them jointly to find the optimal balance between memory usage and concurrency.
# Higher values of `max_tokens_per_forward` increase throughput but use more activation memory, reducing available KV cache.
# Higher values of `max_seqs_per_forward` increase batch size and throughput, but require larger KV cache.

# Since we want to maximize the throughput, we set the batch size to the largest value we can fit in GPU RAM.

MAX_BATCH_SIZE = 1024
MAX_TOKENS_PER_FORWARD = 8192

# ### Torch compile

# We [torch compile](https://github.com/ScalingIntelligence/tokasaurus#torch-compile) the model to make it faster and reduce the amount of used activation memory, allowing us to increase the KV cache size further.
# While this increases server startup time, it's worth it to increase the throughput.

# ### Hydragen

# [Hydragen](https://arxiv.org/abs/2402.05099) (AKA cascade attention, bifurcated attention) improves attention efficiency for [batches of sequences](https://arxiv.org/abs/2402.05099) that share a common prefix.
# You can tune the thresholds where groups will be formed with:
# - `hydragen_min_group_size`: minimum number of sequences in a shared prefix group
# - `hydragen_min_prefix_len`: minimum token length of a shared prefix measured in tokens
# Note that it can have a slight numerical impact on your generations since attention results are combined in bfloat16.

hydragen_min_group_size = 256  # must be > cudagraph_max_size (128)
hydragen_min_prefix_len = 256

# ### Misc

port = 10210  # The port the server listens on. Note that all data parallel replicas are accessed through the same server port.
page_size = 16  # The page size for the paged KV cache.
stop_string_num_token_lookback = 5  # How many tokens to look back in the sequence for when checking whether a stop string has been generated. You may need to increase this if you have very long stop strings.
stats_report_seconds = 5.0  # How often server stats are printed to the console.
uvicorn_log_level = "info"  # The logging level for the uvicorn web server handling requests. Set this value to "warning" to disable logs being printed every time a request is finished (which can sometimes be annoying/verbose).

# ## Serving inference at tens of thousands of tokens per second

# The function below spawns a Tokasaurus instance listening at port 8000, serving requests to our model.
# We wrap it in the [`@modal.web_server` decorator](https://modal.com/docs/guide/webhooks#non-asgi-web-servers)
# to connect it to the Internet.

# The server runs in an independent process, via `subprocess.Popen`, and only starts accepting requests
# once the model is spun up and the `serve` function returns.

app = modal.App(app_name)

N_GPU = 1
MINUTES = 60  # seconds


@app.function(
    image=toka_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=port, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "toka",
        f"model={MODEL_NAME}",
        # "dp_size=1",
        # f"tp_size={N_GPU}",
        # "pp_size=1",
        # f"max_tokens_per_forward={MAX_TOKENS_PER_FORWARD}",
        # f"max_seqs_per_forward={MAX_BATCH_SIZE}",
        # "torch_compile=T",
        # "use_hydragen=T",
        # f"hydragen_min_group_size={hydragen_min_group_size}",
        # f"hydragen_min_prefix_len={hydragen_min_prefix_len}",
        # f"page_size={page_size}",
        # f"stop_string_num_token_lookback={stop_string_num_token_lookback}",
        # f"stats_report_seconds={stats_report_seconds}",
        # f"uvicorn_log_level={uvicorn_log_level}",
    ]

    print(cmd)

    subprocess.Popen(" ".join(cmd), shell=True)


# ## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy tokasaurus_throughput.py
# ```

# This will create a new app on Modal, build the container image for it if it hasn't been built yet,
# and deploy the app.

# ## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-vllm-openai-compatible-serve.modal.run`.

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
# modal run tokasaurus_throughput.py
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
        content = "Explain the singular value decomposition."

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
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
    payload: dict[str, object] = {"messages": messages, "model": model, "stream": True}

    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

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
            print(chunk["choices"][0]["delta"]["content"], end="")
    print()


# We also include a basic example of a load-testing setup using
# `locust` in the `load_test.py` script [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible):

# ```bash
# modal run openai_compatible/load_test.py
# ```
