# ---
# deploy: true
# cmd: ["python", "06_gpu_and_ml/llm-serving/sglang_kitchen_sink.py"]
# ---

# # Fast-booting, low-latency Qwen 3 8B with SGLang, GPU snapshots, and speculative decoding

# This is a bare-bones "kitchen-sink" demo of all of the tips and tricks
# you can use to make Qwen 3 8B go brrt.

# Unlike our other examples, this demo includes limited explanation of the code.
# For a detailed guide to the principles and practices implemented here,
# see [this guide](https://modal.com/docs/guide/high-performance-llm-inference).

# ## Set up the container image

import asyncio
import json
import subprocess
import time

import aiohttp
import modal
import modal.experimental

MINUTES = 60  # seconds

sglang_image = (
    modal.Image.from_registry(
        "modalresearch/sglang:v0.5.7-fa4-dflash-preview"  # bleeding-edge custom SGLang build
    ).entrypoint([])  # silence chatty logs on container start
)

sglang_image.env(
    {  # bleeding-edge SGLang perf opt settings
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_ENABLE_DFLASH_SPEC_V2": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    }
)

# ## Choose a GPU

GPU_TYPE, N_GPUS = "B200", 1
GPU = f"{GPU_TYPE}:{N_GPUS}"

# ### Loading and cacheing the model weights

MODEL_NAME = "Qwen/Qwen3-8B-FP8"
MODEL_REVISION = (  # pin revision id to avoid nasty surprises!
    "220b46e3b2180893580a4454f21f22d3ebb187d3"  # latest commit as of 2026-01-29, from 2025-07-25
)

sglang_image = sglang_image.uv_pip_install("huggingface-hub==0.36.0")

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"
MODEL_PATH = f"{HF_CACHE_PATH}/{MODEL_NAME}"

sglang_image = sglang_image.env(
    {"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"}
)

# ### Cacheing compilation artifacts

# JIT DeepGEMM kernels are on by default, but we explicitly enable them via an environment variable.

DG_CACHE_VOL = modal.Volume.from_name("deepgemm-cache", create_if_missing=True)
DG_CACHE_PATH = "/root/.cache/deepgemm"

sglang_image = sglang_image.env({"SGLANG_ENABLE_JIT_DEEPGEMM": "1"})


def compile_deep_gemm():
    import os

    if int(os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM", "1")):
        subprocess.run(
            f"python3 -m sglang.compile_deep_gemm --model-path {MODEL_NAME} --revision {MODEL_REVISION} --tp {N_GPUS}",
            shell=True,
        )


sglang_image = sglang_image.run_function(
    compile_deep_gemm,
    volumes={DG_CACHE_PATH: DG_CACHE_VOL, HF_CACHE_PATH: HF_CACHE_VOL},
    gpu=GPU,
)

# ## Configure SGLang for minimal latency

speculative_config = {  # use bleeding-edge speculative decoding method
    "speculative-algorithm": "DFLASH",
    "speculative-draft-model-path": "z-lab/Qwen3-8B-DFlash-b16",
}

# ## Speed up cold starts with GPU snapshotting

sglang_image = sglang_image.env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})

# ### Sleeping and waking an SGLang server

with sglang_image.imports():
    import requests


def warmup():
    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=10
        ).raise_for_status()


def sleep():
    requests.post(
        f"http://127.0.0.1:{PORT}/release_memory_occupation", json={}
    ).raise_for_status()


def wake_up():
    requests.post(
        f"http://127.0.0.1:{PORT}/resume_memory_occupation", json={}
    ).raise_for_status()


# ## Define the inference server and infrastructure

# ### Selecting infrastructure to minimize latency

REGION = "us"
PROXY_REGION = "us-west"

MIN_CONTAINERS = 0  # set to 1 to ensure one replica is always ready

# ### Determining autoscaling policy with `@modal.concurrent`

TARGET_INPUTS = 10

# ### Controlling container lifecycles with `modal.Cls`


def wait_ready(process: subprocess.Popen, timeout: int = 5 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(5)
    raise TimeoutError(f"SGLang server not ready within {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


app = modal.App(name="example-sglang-kitchen-sink")
PORT = 8000


@app.cls(
    image=sglang_image,
    gpu=GPU,
    volumes={DG_CACHE_PATH: DG_CACHE_VOL, HF_CACHE_PATH: HF_CACHE_VOL},
    region=REGION,
    min_containers=MIN_CONTAINERS,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.experimental.http_server(
    port=PORT,  # wrapped code must listen on this port
    proxy_regions=[PROXY_REGION],  # location of proxies, should be same as Cls region
    exit_grace_period=15,  # seconds, time to finish up requests when closing down
)
@modal.concurrent(target_inputs=TARGET_INPUTS)
class SGLang:
    @modal.enter(snap=True)
    def startup(self):
        """Start the SGLang server and block until it is healthy, then warm it up and put it to sleep."""
        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            f"{PORT}",
            "--tp",  # use all GPUs to split up tensor-parallel operations
            f"{N_GPUS}",
            "--cuda-graph-max-bs",  # capture CUDA graphs up to batch sizes we're likely to observe
            f"{TARGET_INPUTS * 2}",
            "--max-running-requests",
            f"{TARGET_INPUTS * 4}",
            "--enable-metrics",  # expose metrics endpoints for telemetry
            "--enable-memory-saver",  # enable offload, for snapshotting
            "--enable-weights-cpu-backup",  # enable offload, for snapshotting
            "--decode-log-interval",  # how often to log during decoding, in tokens
            "100",
            "--mem-fraction",  # leave space for speculative model
            "0.8",
            "--attention-backend",
            "fa4",  # use bleeding-edge attention backend
        ]

        cmd += [  # add speculative config
            item for k, v in speculative_config.items() for item in (f"--{k}", str(v))
        ]

        self.process = subprocess.Popen(cmd, start_new_session=True)
        wait_ready(self.process)
        warmup()
        sleep()  # release GPU memory occupation before snapshot

    @modal.enter(snap=False)
    def restore(self):
        """After snapshot restoration, resume GPU memory occupation."""
        wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()


# ## Deploy the server

# ```bash
# modal deploy sglang_kitchen_sink.py
# ```

# ## Test the server

# ```bash
# modal run sglang_kitchen_sink.py
# ```


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, prompt=None, twice=True):
    url = SGLang._experimental_get_flash_urls()[0]

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
        await probe(url, messages, timeout=1 * MINUTES)


async def probe(url, messages=None, timeout=5 * MINUTES):
    if messages is None:
        messages = [
            {
                "role": "user",
                "content": "Write me five very different programs, repeated in Python and Rust.",
            }
        ]

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
        print(full_text)


# ### Test memory snapshotting

# ```bash
# python sglang_kitchen_sink.py
# ```

if __name__ == "__main__":
    # after deployment, we can use the class from anywhere
    SGLang = modal.Cls.from_name("example-sglang-kitchen-sink", "SGLang")

    print("calling inference server")
    try:
        asyncio.run(probe(SGLang._experimental_get_flash_urls()[0]))
    except modal.exception.NotFoundError as e:
        raise Exception(
            f"To take advantage of GPU snapshots, deploy first with modal deploy {__file__}"
        ) from e
