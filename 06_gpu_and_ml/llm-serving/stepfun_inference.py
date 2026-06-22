# # Run StepFun models with SGLang

# In this example, we show how to run a [SGLang](https://github.com/sgl-project/sglang) server
# on Modal serving [StepFun's Step 3.7 Flash](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8).

# ## Set up the container image

import asyncio
import json
import subprocess
import time

import aiohttp
import modal

MINUTES = 60  # seconds

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:dev-cu13-dev-step-3.7-flash")
    .entrypoint([])  # silence chatty logs on container start
    .run_commands("rm -rf /root/.cache/huggingface")  # clean up
)

# We'll need 8 H100 GPUs to run this 196B parameter MoE model.
# 8 GPUs × 80GB = 640GB VRAM, enough for the ~190GB FP8 model with KV cache overhead.

N_GPUS = 8
GPU = f"H100:{N_GPUS}"

# ### Loading and cacheing the model weights

MODEL_NAME = "stepfun-ai/Step-3.7-Flash-FP8"
MODEL_REVISION = "d14f10bf45f025eae0f096ce7c91e9c08b0416da"

# We use a [Modal Volume](https://modal.com/docs/guide/volumes) to cache model weights
# so we don't re-download them on every cold start.

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

# We also include a [Modal Secret](https://modal.com/docs/guide/secrets)
# with Hugging Face API credentials so that we can download the model faster.
# You can create a Secret [here](https://modal.com/secrets).

hf_secret = modal.Secret.from_name("huggingface-secret")

sglang_image = sglang_image.env(
    {"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"}
)

# We'll use the `requests` library to check server health and warm up the model.

with sglang_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, port: int, timeout: int = 10 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{port}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(1)
    raise TimeoutError(f"SGLang server not ready within timeout of {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def warmup(port: int):
    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=120
        ).raise_for_status()


# ## Define the inference server

app = modal.App(name="example-stepfun-inference")
PORT = 8000
TARGET_INPUTS = 16


@app.server(
    image=sglang_image,
    gpu=GPU,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    secrets=[hf_secret],
    scaledown_window=15 * MINUTES,
    startup_timeout=120 * MINUTES,
    routing_region="us-east",
    port=PORT,
    target_concurrency=TARGET_INPUTS,
)
class SGLang:
    @modal.enter()
    def startup(self):
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            f"{PORT}",
            "--tp",
            f"{N_GPUS}",
            "--ep",
            f"{N_GPUS}",
            "--cuda-graph-max-bs",
            f"{TARGET_INPUTS * 2}",
            "--max-running-requests",
            f"{TARGET_INPUTS * 2}",
            "--enable-metrics",
            "--trust-remote-code",
        ]

        cmd += (
            [
                "--revision",
                MODEL_REVISION,
            ]
            if MODEL_REVISION
            else []
        )

        self.process = subprocess.Popen(cmd)
        wait_ready(self.process, PORT)
        warmup(PORT)

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Deploy the server

# To deploy the server on Modal, run:

# ```bash
# modal deploy stepfun_inference.py
# ```

# ## Test the server

# To test the server setup, run:

# ```bash
# modal run stepfun_inference.py
# ```


@app.local_entrypoint()
async def test(test_timeout=40 * MINUTES, prompt=None, twice=True):
    url = await SGLang.get_url()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if prompt is None:
        prompt = "Explain the Singular Value Decomposition."

    content = [{"type": "text", "text": prompt}]

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    await probe(url, messages, timeout=test_timeout)
    if twice:
        messages[0]["content"] = "You are Jar Jar Binks."
        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await probe(url, messages, timeout=test_timeout)


async def probe(url, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [{"role": "user", "content": "Tell me a joke."}]

    client_id = str(0)  # set this to some string per multi-turn interaction
    # often a UUID per "conversation"
    headers = {"Modal-Session-ID": client_id}
    deadline = time.time() + timeout
    async with aiohttp.ClientSession(base_url=url, headers=headers) as session:
        while time.time() < deadline:
            try:
                await _send_request_streaming(session, messages)
                return
            except asyncio.TimeoutError:
                await asyncio.sleep(1)
            except aiohttp.client_exceptions.ClientResponseError as e:
                if e.status == 503:
                    await asyncio.sleep(1)
                    continue
                raise e
    raise TimeoutError(f"No response from server within {timeout} seconds")


async def _send_request_streaming(
    session: aiohttp.ClientSession, messages: list, timeout: int | None = None
) -> None:
    payload = {"messages": messages, "stream": True}
    headers = {"Accept": "text/event-stream"}

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=timeout
    ) as resp:
        resp.raise_for_status()
        full_text = ""

        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Server-Sent Events format: "data: ...."
            if not line.startswith("data:"):
                continue

            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            try:
                evt = json.loads(data)
            except json.JSONDecodeError:
                # ignore any non-JSON keepalive
                continue

            delta = (evt.get("choices") or [{}])[0].get("delta") or {}
            chunk = delta.get("content")

            if chunk:
                print(chunk, end="", flush="\n" in chunk or "." in chunk)
                full_text += chunk
        print()  # newline after stream completes
