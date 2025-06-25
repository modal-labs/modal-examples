# ---
# deploy: true
# ---

# # Serverless Tokasaurus (LLama-3.2-1B-Instruct)

# In this example, we demonstrate how to use the Tokasaurus framework to serve Llama-3.2-1B-Instruct model
# at high throughput, benchmarked by [reproducing an experiment](https://github.com/ScalingIntelligence/tokasaurus/blob/a0155181f09c0cf40783e01a625b041985667a92/tokasaurus/benchmarks/standalone_monkeys_gsm8k.py)
# from Large Language Monkeys, where 128 problems from the GSM8K math dataset with 1024 answers to every problem are generated.
# Using Modal's [concurrent decorator](https://modal.com/docs/guide/concurrent-inputs#enabling-input-concurrency),
# we achieve a throughput of around 370k tokens/second, nearly 5x higher than what the [authors reported](https://scalingintelligence.stanford.edu/blogs/tokasaurus/).

# Sample results from running the benchmark:

# |    k |   pass@k |
# |------|----------|
# |    1 | 0.286522 |
# |    2 | 0.413421 |
# |    3 | 0.492062 |
# |    4 | 0.547901 |
# |    5 | 0.590428 |
# |    6 | 0.624252 |
# |    7 | 0.651968 |
# |    8 | 0.675185 |
# |    9 | 0.694966 |
# |   10 | 0.712049 |
# |  100 | 0.923327 |
# | 1000 | 0.991634 |
# | 1024 | 0.992188 |

# Elapsed time: 248.4 seconds
# Total input tokens: 91574
# Total output tokens: 91941963
# Throughput: 370103.90 tokens/second

# ## Overview

# This guide is intended to document two things:
# the general process for building Tokasaurus on Modal
# and a specific configuration for serving the Llama-3.2-1B-Instruct model.

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# Tokasaurus can be installed with `pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

# We take note of the [CUDA version](https://github.com/ScalingIntelligence/tokasaurus/blob/main/logs/blog_commands.md)
# the authors used to build the tokasaurus image.

import asyncio
import json
import re

import aiohttp
import modal

toka_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode tokasaurus==0.0.2 huggingface_hub[hf_transfer]==0.33.0 datasets==3.6.0 tabulate==0.9.0"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# ## Download the model weights

# We'll be running a fine-tuned instruction-following model -- Llama-3.2-1B-Instruct
# that's trained to chat and follow instructions. Since this is a gated model,
# you'll need to [accept the terms of use](https://huggingface.co/meta-llama/Llama-3.2-1B)
# and create a [Secret](https://modal.com/secrets/)
# with your Hugging Face token to download the weights.

secrets = [modal.Secret.from_name("huggingface-secret")]

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_REVISION = "4e20de362430cd3b72f300e6b0f18e50e7166e08"  # avoid nasty surprises when repos update!

# Although Tokasaurus will download weights from Hugging Face on-demand,
# we want to cache them so we don't do it every time our server starts.
# We'll use [Modal Volumes](https://modal.com/docs/guide/volumes) for our cache.
# Modal Volumes are essentially a "shared disk" that all Modal Functions can access like it's a regular disk.

app_name = "example-tokasaurus-throughput"

hf_cache_vol = modal.Volume.from_name(f"{app_name}-hf-cache", create_if_missing=True)

volumes = {
    "/root/.cache/huggingface": hf_cache_vol,
}

# ## Maximizing throughput

# On throughput-focused benchmarks with high prefix sharing workloads, Tokasaurus can outperform vLLM and SGLang by up to 3x+.
# For small models, it benefits from very low CPU overhead by maintaining a deep input queue,
# and dynamic Hydragen grouping to exploit shared prefixes via a greedy depth-first search algorithm.
# For larger models, it supports async tensor parallelism for GPUs with NVLink and a fast implementation of pipeline parallelism for GPUs without.

# We start by maximizing the number of tokens processed per forward pass by adjusting the following parameters:
# - `kv_cache_num_tokens`: max tokens in the KV cache, higher values increase throughput but increase size of KV cache, reducing available KV cache.
# - `max_tokens_per_forward`: max tokens/seq processed per forward pass, higher values increase throughput but use more activation memory, reducing available KV cache.
# - `max_seqs_per_forward`: max sequences processed per forward pass, higher values increase batch size and throughput, but require larger KV cache.

# Since we want to maximize the throughput, we set the batch size to the largest value we can fit in GPU RAM.

KV_CACHE_NUM_TOKENS = (1024 + 512) * 1024
MAX_TOKENS_PER_FORWARD = 32768
MAX_SEQS_PER_FORWARD = 8192

# We could torch compile the model to make it faster and reduce the amount of used activation memory,
# allowing us to use a larger KV cache. However, it dramatically increases the startup time of the server,
# so we don't use it here.

TORCH_COMPILE = None

# We also use [Hydragen](https://arxiv.org/abs/2402.05099) to more efficiently compute attention over a batch of sequences that share a common prefix.

USE_HYDRAGEN = "T"
HYDRAGEN_MIN_GROUP_SIZE = 129

# And some miscellaneous settings:

PAGE_SIZE = 16
STOP_STRING_NUM_TOKEN_LOOKBACK = 5
STATS_REPORT_SECONDS = 5.0
UVICORN_LOG_LEVEL = "info"

MAX_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 1.0
STOP_STRING = json.dumps(["Question:"])
N = 1024

KS = list(range(1, min(11, N + 1)))
cur = 100
while True:
    KS.append(cur)
    cur *= 10
    if cur > N:
        break
if N not in KS:
    KS.append(N)

# ## Serving inference

# The function below spawns a Tokasaurus instance listening at port 10210, serving requests to our model.
# We wrap it in the [`@modal.web_server` decorator](https://modal.com/docs/guide/webhooks#non-asgi-web-servers)
# to connect it to the Internet.

# The server runs in an independent process, via `subprocess.Popen`, and only starts accepting requests
# once the model is spun up and the `serve` function returns.

app = modal.App(app_name)

N_GPU = 1
GPU_CONFIG = f"H100:{N_GPU}"
MINUTES = 60  # seconds

port = 10210


@app.function(
    image=toka_image,
    gpu=GPU_CONFIG,
    scaledown_window=60 * MINUTES,  # how long should we stay up with no requests?
    timeout=60 * MINUTES,  # how long should we wait for container start?
    volumes=volumes,
    secrets=secrets,
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=4
)
@modal.web_server(port=port, startup_timeout=60 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "tksrs",
        f"model={MODEL_NAME}",
        f"kv_cache_num_tokens={KV_CACHE_NUM_TOKENS}",
        f"max_seqs_per_forward={MAX_SEQS_PER_FORWARD}",
        f"max_tokens_per_forward={MAX_TOKENS_PER_FORWARD}",
        f"torch_compile={TORCH_COMPILE}" if TORCH_COMPILE is not None else None,
        f"use_hydragen={USE_HYDRAGEN}",
        f"hydragen_min_group_size={HYDRAGEN_MIN_GROUP_SIZE}",
        f"page_size={PAGE_SIZE}",
        f"stop_string_num_token_lookback={STOP_STRING_NUM_TOKEN_LOOKBACK}",
        f"stats_report_seconds={STATS_REPORT_SECONDS}",
        f"uvicorn_log_level={UVICORN_LOG_LEVEL}",
    ]

    print(" ".join(cmd))

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
# something like `https://<your-workspace-name>--example-tokasaurus-throughput-serve.modal.run`.

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
# that measures the throughput of the server.
# For simplicity, we load the FineWeb-Edu 10BT sample dataset and sample 512 text chunks from it.
# Then, we'll randomly sample seven distinct tasks (added as prefixes).
# These prefixes ensure KV cache misses for the remainder of the generations,
# to keep the benchmark closer to what can be expected in a real workload.

# If you execute the command

# ```bash
# modal run tokasaurus_throughput.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.function(image=toka_image, volumes=volumes)
def load_dataset():
    from datasets import load_dataset

    raw_test_dataset = list(load_dataset("gsm8k", "main", split="test"))
    train_dataset = list(load_dataset("gsm8k", "main", split="train"))
    return raw_test_dataset, train_dataset


def _make_prompt(item: dict) -> str:
    few_shot_items = item["few_shot_items"]
    few_shot_pieces = []
    for f in few_shot_items:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/568af943e315100af3f00937bfd6947844769ab8/lm_eval/tasks/gsm8k/gsm8k.yaml
        few_shot_prompt = f"Question: {f['question']}\nAnswer: {f['answer']}\n\n"
        few_shot_pieces.append(few_shot_prompt)
    few_shot_prompt = "".join(few_shot_pieces)
    prompt = few_shot_prompt + f"Question: {item['question']}\nAnswer:"
    return prompt


async def _send_request(session: aiohttp.ClientSession, prompt: str) -> list[str]:
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
        "/v1/completions", json=payload, headers=headers, timeout=60 * MINUTES
    ) as resp:
        resp.raise_for_status()
        resp_json = await resp.json()
        return [choice["text"] for choice in resp_json["choices"]]


@app.function(image=toka_image)
def tablify(corrects_list: list[list[bool]]):
    import numpy as np
    from tabulate import tabulate

    table = []

    def pass_at_k(n, c, k):
        """
        :param n: total number of samples
        :param c: number of correct samples
        :param k: k in pass@$k$
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    for k in KS:
        to_mean = []
        for corrects in corrects_list:
            to_mean.append(pass_at_k(n=len(corrects), c=sum(corrects), k=k))
        table.append([k, np.mean(to_mean)])

    print(tabulate(table, headers=["k", "pass@k"], tablefmt="github"))


@app.function(image=toka_image, volumes=volumes)
def count_input_tokens(text: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return len(tokenizer.encode(text))


@app.function(image=toka_image, volumes=volumes)
def count_output_tokens(completions: list[str]) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inner_tokenizer = tokenizer._tokenizer
    enc_out = inner_tokenizer.encode_batch(completions)
    n_out_tokens = sum(len(out) for out in enc_out)
    return n_out_tokens


@app.local_entrypoint()
async def test(
    seed: int = 42,
    limit: int = 128,
    num_few_shot: int = 4,
    reps: int = 1,
):
    import random
    import time

    raw_test_dataset, train_dataset = load_dataset.remote()
    print(f"Number of test items: {len(raw_test_dataset)}")
    print(f"Number of train items: {len(train_dataset)}")

    random.seed(seed)

    for i, data in enumerate(train_dataset):
        data["index"] = i

    for i, data in enumerate(raw_test_dataset):
        data["index"] = i
    random.shuffle(raw_test_dataset)
    for i, data in enumerate(raw_test_dataset):
        data["shuffled_index"] = i
    if limit is not None:
        limit = limit
    else:
        limit = len(raw_test_dataset)
    test_dataset = raw_test_dataset[:limit]
    for i, data in enumerate(test_dataset):
        few_shot_items = random.sample(train_dataset, num_few_shot)
        data["few_shot_items"] = few_shot_items
    print(f"Total number of items to process: {len(test_dataset)}")

    url = serve.get_web_url()
    for _ in range(reps):
        start = time.time()

        async def process_item(item):
            async with aiohttp.ClientSession(base_url=url) as session:
                return await _send_request(session, _make_prompt(item))

        tasks = [process_item(item) for item in test_dataset]
        completions_list = await asyncio.gather(*tasks)
        for completions in completions_list:
            assert len(completions) == N

        ANS_RE_GSM8k = re.compile(r"#### (\-?[\$0-9\.\,]+)")
        INVALID_ANS_GSM8k = "[invalid]"
        GSM8K_IGNORE_REGEXES = [",", "\\$", "\\.$"]

        def extract_answer_gsm8k(st):
            match = ANS_RE_GSM8k.search(st)
            if match:
                match_str = match.group(1).strip()
                if GSM8K_IGNORE_REGEXES is not None:
                    for s in GSM8K_IGNORE_REGEXES:
                        match_str = re.sub(s, "", match_str)
                return match_str
            else:
                return INVALID_ANS_GSM8k

        lst_results = []
        for item, completions in zip(test_dataset, completions_list):
            corrects = []
            for completion in completions:
                gt_answer = extract_answer_gsm8k(item["answer"])
                assert gt_answer != INVALID_ANS_GSM8k
                corrects.append(extract_answer_gsm8k(completion) == gt_answer)
            result = {
                "prompt": _make_prompt(item),
                "completions": completions,
                "corrects": corrects,
            }
            lst_results.append(result)

        end = time.time()

        elapsed = end - start
        print(f"Elapsed time: {elapsed} seconds")

        corrects_list = [result["corrects"] for result in lst_results]
        tablify.remote(corrects_list)

        total_input_tokens = 0
        async for count in count_input_tokens.map.aio(
            [result["prompt"] for result in lst_results]
        ):
            total_input_tokens += count

        total_output_tokens = 0
        async for count in count_output_tokens.map.aio(
            [result["completions"] for result in lst_results]
        ):
            total_output_tokens += count

        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")

        throughput = total_output_tokens / elapsed
        print(f"Throughput: {throughput:.2f} tokens/second")


# We also include a basic example of a load-testing setup using
# `locust` in the `load_test.py` script [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible):

# ```bash
# modal run openai_compatible/load_test.py
# ```
