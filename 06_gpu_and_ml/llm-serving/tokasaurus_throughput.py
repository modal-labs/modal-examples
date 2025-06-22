# ---
# deploy: true
# ---

# # Serverless Tokasaurus (Qwen2-7B-Instruct)

# In this example, we demonstrate how to use the Tokasaurus framework to serve Qwen2-7B-Instruct model
# at high throughput for structured output extraction from the FineWeb-Edu 10BT sample dataset.

# ## Overview

# This guide is intended to document two things:
# the general process for building Tokasaurus on Modal
# and a specific configuration for serving the Qwen2-7B-Instruct model.

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# Tokasaurus can be installed with `pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

# We take note of the [CUDA version](https://github.com/ScalingIntelligence/tokasaurus/blob/main/logs/blog_commands.md)
# the authors used to build the tokasaurus image.

import aiohttp
import modal

toka_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode tokasaurus==0.0.2 huggingface_hub[hf_transfer]==0.33.0 datasets==3.6.0"
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

volumes = {
    "/root/.cache/huggingface": hf_cache_vol,
}

# ## Maximizing throughput

# On throughput-focused benchmarks with high prefix sharing workloads, Tokasaurus can outperform vLLM and SGLang by up to 3x+.
# For small models, it benefits from very low CPU overhead by maintaining a deep input queue,
# and dynamic Hydragen grouping to exploit shared prefixes via a greedy depth-first search algorithm.
# For larger models, it supports async tensor parallelism for GPUs with NVLink and a fast implementation of pipeline parallelism for GPUs without.

# We start by maximizing the number of tokens processed per forward pass by adjusting the following two parameters:
# - `max_tokens_per_forward`: max tokens processed per forward pass, higher values increase throughput but use more activation memory, reducing available KV cache.
# - `max_seqs_per_forward`: max sequences processed per forward pass, higher values increase batch size and throughput, but require larger KV cache.

# Since we want to maximize the throughput, we set the batch size to the largest value we can fit in GPU RAM.

MAX_TOKENS_PER_FORWARD = 2**17
MAX_SEQS_PER_FORWARD = 512

# ## Serving inference

# The function below]spawns a Tokasaurus instance listening at port 10210, serving requests to our model.
# We wrap it in the [`@modal.web_server` decorator](https://modal.com/docs/guide/webhooks#non-asgi-web-servers)
# to connect it to the Internet.

# The server runs in an independent process, via `subprocess.Popen`, and only starts accepting requests
# once the model is spun up and the `serve` function returns.

app = modal.App(app_name)

N_GPU = 1
GPU_CONFIG = f"H200:{N_GPU}"
MINUTES = 60  # seconds

port = 10210


@app.function(
    image=toka_image,
    gpu=GPU_CONFIG,
    scaledown_window=60 * MINUTES,  # how long should we stay up with no requests?
    timeout=60 * MINUTES,  # how long should we wait for container start?
    volumes=volumes,
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=min(MAX_SEQS_PER_FORWARD, 1000)  # modal max is 1000
)
@modal.web_server(port=port, startup_timeout=60 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "toka",
        f"model={MODEL_NAME}",
        "dp_size=1",
        f"tp_size={N_GPU}",
        "pp_size=1",
        f"max_tokens_per_forward={MAX_TOKENS_PER_FORWARD}",
        f"max_seqs_per_forward={MAX_SEQS_PER_FORWARD}",
        "page_size=16",  # The page size for the paged KV cache
        "stop_string_num_token_lookback=5",  # How many tokens to look back for stop string detection
        "stats_report_seconds=5.0",
        "uvicorn_log_level=info",
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
def load_dataset(n_samples: int):
    from datasets import load_dataset

    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    samples = []
    for chunk in fw:
        samples.append(chunk["text"])
        if len(samples) >= n_samples:
            break
    return samples


@app.function(image=toka_image, volumes=volumes)
def count_tokens(text: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return len(tokenizer.encode(text))


@app.local_entrypoint()
async def test():
    import asyncio
    import random
    import textwrap
    import time

    texts = load_dataset.remote(n_samples=MAX_SEQS_PER_FORWARD)
    tasks = [
        "Extract key information and return as JSON with fields: title, main_topic, key_points (array), difficulty_level (beginner/intermediate/advanced), estimated_reading_time_minutes\n\n",
        "Analyze the content and return as YAML with sections: summary (2-3 sentences), learning_objectives (list), prerequisites (list), related_concepts (list), assessment_questions (list of 3 questions)\n\n",
        "Create a structured response in XML format with tags: <content_type>, <target_audience>, <core_concepts>, <practical_applications>, <further_reading>\n\n",
        "Return a JSON object with: topic_category, complexity_score (1-10), essential_vocabulary (array), step_by_step_breakdown (array), common_misconceptions (array)\n\n",
        "Format as TOML with sections: [metadata] (title, author, date), [content_analysis] (main_theme, subtopics, difficulty), [educational_value] (skills_developed, real_world_relevance, prerequisites)\n\n",
        "Generate a JSON response with: subject_area, learning_outcomes (array), time_requirements, skill_level, interactive_elements (array), assessment_criteria (object with rubric fields)\n\n",
        "Return structured data as YAML with: educational_framework (standards_aligned, grade_level), content_structure (sections, key_terminology), pedagogical_approach (teaching_methods, student_activities), evaluation_metrics (assessment_types, success_criteria)\n\n",
    ]
    # prepending any string that causes a tokenization change is enough to invalidate KV cache
    sampled_tasks = random.choices(tasks, k=len(texts))
    questions = [task + text for task, text in zip(sampled_tasks, texts)]

    url = serve.get_web_url()

    system_prompt = {
        "role": "system",
        "content": textwrap.dedent("""
        You are an expert at extracting structured data from text.
        You will be given a text and a task.
        You will need to return the text in the format specified by the task.
        """),
    }
    messages = [
        [  # OpenAI chat format
            system_prompt,
            {"role": "user", "content": question},
        ]
        for question in questions
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Generating completions for batch of size {len(messages)}...")
        start = time.monotonic_ns()

        tasks = []
        for msg_set in messages:
            tasks.append(_send_request(session, msg_set))
        responses = await asyncio.gather(*tasks)

        duration_s = (time.monotonic_ns() - start) / 1e9

        token_counts = []
        async for count in count_tokens.map.aio(responses):
            token_counts.append(count)
        num_tokens = sum(token_counts)

        print(
            f"Generated {num_tokens} tokens from {MODEL_NAME} in {duration_s:.1f} seconds,"
            f" throughput = {num_tokens / duration_s:.0f} tokens/second for batch of size {len(messages)} on {GPU_CONFIG}."
        )


async def _send_request(session: aiohttp.ClientSession, messages: list) -> str:
    payload: dict[str, object] = {
        "messages": messages,
        "model": "llm",
        "max_tokens": 8192,
        "n": 1,
        "temperature": 0.0,
    }

    headers = {"Content-Type": "application/json"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=60 * MINUTES
    ) as resp:
        resp.raise_for_status()
        resp_json = await resp.json()
        return resp_json["choices"][0]["message"]["content"]


# We also include a basic example of a load-testing setup using
# `locust` in the `load_test.py` script [here](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving/openai_compatible):

# ```bash
# modal run openai_compatible/load_test.py
# ```
