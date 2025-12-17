# ---
# deploy: true
# cmd: ["python", "06_gpu_and_ml/llm-serving/sglang_low_latency.py"]
# ---

# # Low Latency Qwen 3-8B with SGLang and Modal

# In this example, we show how to serve Qwen 3-8B with SGLang on Modal using @modal.experimental.http_server.
# This is a new low latency routing service on Modal which offers significantly reduced overheads, higher throughput, and session based routing.
# These features make `http_server` especially useful for inference workloads.

# We also include instructions for cutting cold start times by an order of magnitude using Modal's [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# We'll use the [SGLang inference server](https://github.com/sgl-project/sglang).
# Note that we need to build the SGLang image from source since the official image does not support the `--enable-cpu-backup` flag.

import json
import subprocess
import time
from typing import Any

import aiohttp
import modal
import modal.experimental

APP_NAME = "example-sglang-low-latency"
MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = (
    "b968826d9c46dd6066d109eabc6255188de91218"  # Latest commit as of 2025-12-16
)

PORT = 8000
MIN_CONTAINERS = 1
MINUTE = 60

HF_CACHE_VOL: modal.Volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
HF_CACHE_PATH: str = "/root/.cache/huggingface"
MODEL_PATH: str = f"{HF_CACHE_PATH}/{MODEL_NAME}"
sglang_image: modal.Image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .uv_pip_install(
        "huggingface-hub==0.36.0",
    )
    .run_commands("git clone https://github.com/sgl-project/sglang.git /sglang")
    .run_commands("pip uninstall -y sglang")
    .run_commands("cd /sglang && pip install -e python[all]")
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TMS_INIT_ENABLE_CPU_BACKUP": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torch/",
        }
    )
)

with sglang_image.imports():
    import requests

app = modal.App(name=APP_NAME)


@app.cls(
    image=sglang_image,
    gpu="H100",
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    region="us-east",
    min_containers=MIN_CONTAINERS,
    timeout=4 * MINUTE,
)
@modal.experimental.http_server(
    port=PORT, proxy_regions=["us-east"], exit_grace_period=5
)
@modal.concurrent(target_inputs=20)
class SGLang:
    """Serve a HuggingFace model via SGLang with readiness check."""

    @modal.enter(snap=True)
    def startup(self) -> None:
        """Start the SGLang server and block until it is healthy."""

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
            "--enable-metrics",
            "--enable-memory-saver",
            "--enable-weights-cpu-backup",
        ]

        self.process = subprocess.Popen(cmd)
        self._wait_ready(self.process)
        self._warmup()
        self._sleep()

    @modal.enter(snap=False)
    def wake_up(self):
        self._wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()

    @staticmethod
    def _warmup():
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "max_tokens": 16,
        }
        for _ in range(3):
            requests.post(
                f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=10
            ).raise_for_status()

    @staticmethod
    def _wait_ready(process: subprocess.Popen, timeout: int = 3 * MINUTE):
        def check_process_is_running() -> Exception | None:
            if process is not None and process.poll() is not None:
                return Exception(
                    f"Process {process.pid} exited with code {process.returncode}"
                )
            return None

        deadline: float = time.time() + timeout
        while time.time() < deadline:
            try:
                if error := check_process_is_running():
                    raise error
                response = requests.get(f"http://127.0.0.1:{PORT}/health")
                if response.status_code == 200:
                    print("Server is healthy")
                    break
            except Exception:
                pass

    @staticmethod
    def _sleep():
        headers = {"Content-Type": "application/json"}
        requests.post(
            f"http://127.0.0.1:{PORT}/release_memory_occupation",
            headers=headers,
            json={},
        ).raise_for_status()

    @staticmethod
    def _wake_up():
        headers = {"Content-Type": "application/json"}
        requests.post(
            f"http://127.0.0.1:{PORT}/resume_memory_occupation",
            headers=headers,
            json={},
        ).raise_for_status()


## Deploy the server

# To deploy the API on Modal, just run
# ```bash
# modal deploy sglang_low_latency.py
# ```

# This will create a new app on Modal, build the container image for it if it hasn't been built yet,
# and deploy the app.

## Interact with the server

# Once it is deployed, you'll see a URL appear in the command line,
# something like `https://your-workspace-name--example-sglang-low-latency-serve.us-east.modal.direct`.

# You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-sglang-low-latency-serve.us-east.modal.direct/docs`.
# These docs describe each route and indicate the expected input and output
# and translate requests into `curl` commands.

# For simple routes like `/health`, which checks whether the server is responding,
# you can even send a request directly from the docs.

## Test the server

# To make it easier to test the server setup, we also include a `local_entrypoint`
# that does a healthcheck and then hits the server.

# If you execute the command

# ```bash
# modal run sglang_low_latency.py
# ```

# a fresh replica of the server will be spun up on Modal while
# the code below executes on your local machine.

# Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# block of a Python script, but for cloud deployments!


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTE, content=None, twice=True):
    url = SGLang._experimental_get_flash_urls()[0]

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

        content = [
            {
                "type": "text",
                "text": "What action do you think I should take in this situation?"
                " List all the possible actions and explain why you think they are good or bad.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

    messages = [  # OpenAI chat format
        system_prompt,
        {"role": "user", "content": content},
    ]

    start_time = time.time()
    async with aiohttp.ClientSession(base_url=url) as session:
        while time.time() - start_time < test_timeout:
            print(f"Running health check for server at {url}")
            async with session.get(
                "/health", timeout=test_timeout - 1 * MINUTE
            ) as resp:
                if resp.status == 200:
                    print(f"Successful health check for server at {url}")
                    break
                time.sleep(10)
        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages, timeout=1 * MINUTE)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list, timeout: int
) -> None:
    # `stream=True` tells an OpenAI-compatible backend to stream chunks
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        "temperature": 0.15,
    }

    # To use sticky routing, set X-Modal-Upstream header. TODO(claudia): update this header to be `Modal-Session-Id`.
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Modal-Upstream": "userA",
    }

    async with session.post(
        "/v1/chat/completions", json=payload, headers=headers, timeout=timeout
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


if __name__ == "__main__":
    import asyncio

    # after deployment, we can use the class from anywhere
    sglang_server = modal.Cls.from_name("example-sglang-low-latency", "SGLang")

    async def test(url):
        messages = [{"role": "user", "content": "Tell me a joke."}]
        async with aiohttp.ClientSession(base_url=url) as session:
            await _send_request(session, MODEL_NAME, messages, timeout=10 * MINUTE)

    print("calling inference server")
    asyncio.run(test(sglang_server._experimental_get_flash_urls()[0]))

