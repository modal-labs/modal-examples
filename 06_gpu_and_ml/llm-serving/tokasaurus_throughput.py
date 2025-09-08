# ---
# args: ["--limit", "2"]
# ---

# # High-throughput LLM inference with Tokasaurus (LLama 3.2 1B Instruct)

# In this example, we demonstrate how to use Tokasaurus, an LLM inference framework designed for maximum throughput.

# It maps the [Large Language Monkeys GSM8K demo](https://github.com/ScalingIntelligence/tokasaurus/blob/a0155181f09c0cf40783e01a625b041985667a92/tokasaurus/benchmarks/standalone_monkeys_gsm8k.py)
# from the [Tokasaurus release blog post](https://scalingintelligence.stanford.edu/blogs/tokasaurus/) onto Modal
# and replicates the core result: sustained inference at >80k tok/s throughput,
# exceeding their reported numbers for vLLM and SGLang by ~3x.

# In the "Large Language Monkeys" inference-time compute scaling paradigm,
# [also introduced by the same Stanford labs](https://arxiv.org/abs/2407.21787),
# the response quality of a system using a small model is improved to match or exceed a system using a large model
# by running many requests in parallel.
# Here, it's applied to the Grade School Math (GSM8K) dataset.

# For more on this LLM inference pattern
# (and an explainer on why it's such a natural fit for current parallel computing systems)
# see [our blog post reproducing and extending their results](https://modal.com/blog/llama-human-eval).

# ## Set up the container image

# Our first order of business is to define the environment our LLM engine will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).

# We translate the [recipe](https://github.com/ScalingIntelligence/tokasaurus/blob/main/logs/blog_commands.md)
# the authors used to build their Tokasaurus environment into methods of `modal.Image`.

# This requires, for instance, picking a base Image that includes the right version of the
# [CUDA toolkit](https://modal.com/gpu-glossary/host-software/cuda-software-platform).

import random
import time

import aiohttp
import modal

toka_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12"
    ).entrypoint([])  # silence chatty logs on container start
)

# We also set an environment variable that directs Torch-based libraries to only compile kernels for the
# [GPU SM architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
# we are targeting, Hopper. This isn't strictly necessary, but it silences some paranoid logs.

GPU_CONFIG = "H100!"  # ! means "strictly", no upgrades to H200
TORCH_CUDA_ARCH_LIST = "9.0 9.0a"  # Hopper, aka H100/H200

# From there, Tokasaurus can be installed like any normal Python package,
# since Modal [provides the host CUDA drivers](https://modal.com/docs/guide/cuda).

toka_image = toka_image.env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1", "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST}
).uv_pip_install(
    "tokasaurus==0.0.2",
    "huggingface_hub[hf_transfer]==0.33.0",
    "datasets==3.6.0",
)

# ## Download the model weights

# For this demo, we run Meta's Llama 3.2 1B Instruct model, downloaded from Hugging Face.
# Since this is a gated model, you'll need to
# [accept the terms of use](https://huggingface.co/meta-llama/Llama-3.2-1B)
# and create a [Secret](https://modal.com/secrets/)
# with your Hugging Face token to download the weights.

secrets = [modal.Secret.from_name("huggingface-secret")]

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"  # avoid nasty surprises when repos update!

# Although Tokasaurus will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use a [Modal Volume](https://modal.com/docs/guide/volumes) for our cache. For more on storing model weights on Modal, see
# [this guide](https://modal.com/docs/guide/model-weights).


app_name = "example-tokasaurus-throughput"
hf_cache_vol = modal.Volume.from_name(f"{app_name}-hf-cache", create_if_missing=True)
volumes = {"/root/.cache/huggingface": hf_cache_vol}

# ## Configure Tokasaurus for maximum throughput on this workload

# On throughput-focused benchmarks with high prefix sharing workloads, Tokasaurus can outperform vLLM and SGLang nearly three-fold.

# For small models like the one we are running, it reduces CPU overhead by maintaining a deep input queue
# and exposing shared prefixes to the GPU [Tensor Cores](https://modal.com/gpu-glossary/device-hardware/tensor-core)
# with [Hydragen](https://arxiv.org/abs/2402.05099).

USE_HYDRAGEN = "T"
HYDRAGEN_MIN_GROUP_SIZE = 129  # sic

# We start by maximizing the number of tokens processed per forward pass by adjusting the following parameters:

# - `kv_cache_num_tokens`: max tokens in the KV cache, higher values increase throughput but consume GPU memory
# - `max_tokens_per_forward`: max tokens/seq processed per forward pass, higher values increase throughput but use more GPU memory
# - `max_seqs_per_forward`: max sequences processed per forward pass, higher values increase batch size and throughput, but require more GPU memory

# We also set a few other parameters with less obvious impacts -- the KV cache page size and the stop token behavior.
# All values are derived from
# [this version of the official benchmarking script](https://github.com/ScalingIntelligence/tokasaurus/blob/a0155181f09c0cf40783e01a625b041985667a92/tokasaurus/benchmarks/standalone_monkeys_gsm8k.py),
# except the `KV_CACHE_NUM_TOKENS`, which we increase to the maximum the GPU can handle.
# The value in the script is set to `(1024 + 512) * 1024`, which is the maximum that the other engines can handle, lower than that of Tokasaurus.

KV_CACHE_NUM_TOKENS = (1024 + 768) * 1024  # tuned for H100, 80 GB RAM
MAX_TOKENS_PER_FORWARD = 32768
MAX_SEQS_PER_FORWARD = 8192
PAGE_SIZE = 16
STOP_STRING_NUM_TOKEN_LOOKBACK = 5

# We could apply the Torch compiler to the model to make it faster and, via kernel fusion, reduce the amount of used activation memory,
# leaving space for a larger KV cache. However, it dramatically increases the startup time of the server,
# and we only see modest (20%, not 2x) improvements to throughput, so we don't use it here.

TORCH_COMPILE = "F"

# Lastly, we need to set a few of the parameters for the client requests,
# again based on the official benchmarking script.

MAX_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 1.0
STOP_STRING = "Question:"
N = 1024

# ## Serve Tokasaurus with an OpenAI-compatible API

# The function below spawns a Tokasaurus instance listening at port `10210`,
# serving an OpenAI-compatible API.
# We wrap it in the [`@modal.web_server` decorator](https://modal.com/docs/guide/webhooks#non-asgi-web-servers)
# to connect it to the Internet.

# The server runs in an independent process, via `subprocess.Popen`.
# If it hasn't started listening on the `PORT` within the `startup_timeout`,
# the server start will be marked as failed.

app = modal.App(app_name)

MINUTES = 60  # seconds
PORT = 10210


@app.function(
    image=toka_image,
    gpu=GPU_CONFIG,
    scaledown_window=60 * MINUTES,  # how long should we stay up with no requests?
    timeout=60 * MINUTES,  # how long should we allow requests to take?
    # long, because we're doing batched inference
    volumes=volumes,
    secrets=secrets,
)
@modal.concurrent(max_inputs=1000)
@modal.web_server(port=PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = " ".join(
        [
            "tksrs",
            f"model={MODEL_NAME}",
            f"kv_cache_num_tokens={KV_CACHE_NUM_TOKENS}",
            f"max_seqs_per_forward={MAX_SEQS_PER_FORWARD}",
            f"max_tokens_per_forward={MAX_TOKENS_PER_FORWARD}",
            f"torch_compile={TORCH_COMPILE}",
            f"use_hydragen={USE_HYDRAGEN}",
            f"hydragen_min_group_size={HYDRAGEN_MIN_GROUP_SIZE}",
            f"stop_string_num_token_lookback={STOP_STRING_NUM_TOKEN_LOOKBACK}",
            "page_size=16",
            "stats_report_seconds=5.0",
            "uvicorn_log_level=info",
        ]
    )

    print(cmd)

    subprocess.Popen(cmd, shell=True)


# The code we have so far is enough to deploy Tokasaurus on Modal.
# Just run:

# ```bash
# modal deploy tokasaurus_throughput.py
# ```

# And you can hit the server with your favorite OpenAI-compatible API client,
# like the `openai` Python SDK.

# ## Run the Large Language Monkeys GSM8K benchmark

# To make it easier to check the performance and to provide a simple test
# that can be used when setting up/configuring a Tokasaurus deployment,
# we include a simple `benchmark` function that acts as a `local_entrypoint`.
# If you target this script with `modal run`, this code will execute,
# spinning up a new replica and sending some test requests to it.

# Because the API responses don't include token counts, we need a quick helper function to
# calculate token counts from a prompt or completion.
# We add [automatic dynamic batching](https://modal.com/docs/guide/dynamic-batching)
# with `modal.batched`, so that we can send single strings but still take advantage
# of batched encoding.


@app.function(image=toka_image, volumes=volumes)
@modal.batched(max_batch_size=128, wait_ms=100)
def count_tokens(texts: list[str]) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return [len(ids) for ids in tokenizer(texts)["input_ids"]]


# You can run the benchmark with

# ```bash
# modal run tokasaurus_throughput.py
# ```

# or pass the `--help` flag to see options.


@app.local_entrypoint()
async def benchmark(seed: int = 42, limit: int = 16, num_few_shot: int = 4):
    import asyncio

    print("Loading dataset")
    dataset = load_dataset.remote(seed=seed, num_few_shot=num_few_shot, limit=limit)
    print(f"Total number of items to process: {len(dataset)}")

    serve.update_autoscaler(
        max_containers=1  # prevent concurrent execution when benchmarking
    )

    url = serve.get_web_url()
    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")

        async with session.get("/v1/models", timeout=20 * MINUTES) as resp:
            up = (  # expect 404, /v1/models not implemented in toka 0.0.2
                resp.status < 500
            )

        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print("Beginning throughput test")
        start = time.time()

        reqs, resps = [], []
        reqs = [_send_request(session, _make_prompt(item)) for item in dataset]
        resps = await asyncio.gather(*reqs)

        end = time.time()
        total_time = end - start
        print(f"Finished throughput test in {int(total_time)}s")

        # sniff test the results
        _integrity_check(resps)

        # calculate throughput from total elapsed time and total token counts
        print("Counting tokens")

        input_text = [resp["prompt"] for resp in resps]
        output_text = [  # flatten completions from list inside a list down to a single list
            completion for resp in resps for completion in resp["completions"]
        ]
        total_tokens = sum(
            [count async for count in count_tokens.map.aio(input_text + output_text)]
        )

        total_throughput = total_tokens // total_time

        print(f"Total throughput: {total_throughput} tokens/second")


# ## Addenda

# The remaining code in this example is utility code, mostly for managing
# the GSM8K dataset. That code is slightly modified from the code in the Tokasaurus repo
# [here](https://github.com/ScalingIntelligence/tokasaurus/blob/a0155181f09c0cf40783e01a625b041985667a92/tokasaurus/benchmarks/standalone_monkeys_gsm8k.py).


@app.function(image=toka_image, volumes=volumes)
def load_dataset(seed: int, num_few_shot: int, limit: int = None):
    from datasets import load_dataset

    test_dataset = list(load_dataset("gsm8k", "main", split="test"))

    random.seed(seed)
    random.shuffle(test_dataset)

    if limit is not None:
        test_dataset = test_dataset[:limit]

    if num_few_shot > 0:
        train_dataset = list(load_dataset("gsm8k", "main", split="train"))
        for i, data in enumerate(test_dataset):
            few_shot_items = random.sample(train_dataset, num_few_shot)
            data["few_shot_items"] = few_shot_items

    return test_dataset


def _make_prompt(item: dict) -> str:
    few_shot_items = item["few_shot_items"]
    few_shot_pieces = []
    for f in few_shot_items:
        few_shot_prompt = f"Question: {f['question']}\nAnswer: {f['answer']}\n\n"
        few_shot_pieces.append(few_shot_prompt)
    few_shot_prompt = "".join(few_shot_pieces)
    prompt = few_shot_prompt + f"Question: {item['question']}\nAnswer:"
    return prompt


def _integrity_check(responses):
    for ii, resp in enumerate(responses):
        n_completions = len(resp["completions"])
        assert n_completions == N, (
            f"Expected {N} completions, got {n_completions} for request {ii}"
        )


async def _send_request(session: aiohttp.ClientSession, prompt: str):
    payload: dict[str, object] = {
        "model": "llm",
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stop": STOP_STRING,
        "n": N,
        "logprobs": None,
    }
    headers = {"Content-Type": "application/json"}

    async with session.post(
        "/v1/completions", json=payload, headers=headers, timeout=10 * MINUTES
    ) as resp:
        resp.raise_for_status()
        resp_json = await resp.json()
        return {
            "prompt": prompt,
            "completions": [choice["text"] for choice in resp_json["choices"]],
        }
