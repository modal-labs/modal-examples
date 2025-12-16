# ---
# deploy: true
# cmd: ["python", "06_gpu_and_ml/llm-serving/vllm_low_latency.py"]
# ---

# # Low Latency Qwen 3-8B with vLLM and Modal

# In this example, we show how to serve Qwen 3-8B with vLLM on Modal using @modal.experimental.http_server.
# This is a new low latency routing service on Modal which offers significantly reduced overheads, higher throughput, and session based routing.
# These features make `http_server` especially useful for inference workloads.

# We also include instructions for cutting cold start times by an order of magnitude using Modal's [CPU + GPU memory snapshots](https://modal.com/docs/guide/memory-snapshot).

# ## Set up the container image

# Our first order of business is to define the environment our server will run in:
# the [container `Image`](https://modal.com/docs/guide/custom-container).
# We'll use the [vLLM inference server](https://docs.vllm.ai).
# vLLM can be installed with `uv pip`, since Modal [provides the CUDA drivers](https://modal.com/docs/guide/cuda).

import subprocess
import time

import modal
import modal.experimental

APP_NAME = "example-vllm-low-latency"
MODEL_NAME = "Qwen/Qwen3-8B"
MODEL_REVISION = (
    "b968826d9c46dd6066d109eabc6255188de91218"  # Latest commit as of 2025-12-16
)

PORT = 8000
MIN_CONTAINERS = 1

HF_CACHE_VOL: modal.Volume = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
HF_CACHE_PATH: str = "/root/.cache/huggingface"
MODEL_PATH: str = f"{HF_CACHE_PATH}/{MODEL_NAME}"
vllm_image: modal.Image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .uv_pip_install("vllm==0.11.2", "huggingface-hub==0.36.0")
    .env(
        {
            "HF_HUB_CACHE": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCH_CPP_LOG_LEVEL": "FATAL",
        }
    )
)

with vllm_image.imports():
    import requests

app = modal.App(name=APP_NAME)

@app.cls(
    image=vllm_image,
    gpu="H100",
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    region="us-east",
    min_containers=MIN_CONTAINERS,
    timeout=6 * 60,
)
@modal.experimental.http_server(
    port=PORT, proxy_regions=["us-east"], exit_grace_period=5
)
@modal.concurrent(target_inputs=20)
class VLLM:
    """Serve a HuggingFace model via VLLM with readiness check."""

    @modal.enter(snap=True)
    def startup(self) -> None:
        """Start the VLLM server and block until it is healthy."""

        cmd: list[str] = [
            "vllm",
            "serve",
            "--uvicorn-log-level",
            "error",
            MODEL_NAME,
            "--revision",
            MODEL_REVISION,
            "--served-model-name",
            MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            f"{PORT}",
            "--disable-uvicorn-access-log",
            "--disable-log-requests",
            "--enable-sleep-mode",
        ]

        self.process = subprocess.Popen(cmd)
        self._wait_ready(self.process)
        self._warmup()
        self._sleep(1)

    @modal.enter(snap=False)
    def wake_up(self):
        self._wake_up()

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
    def _wait_ready(process: subprocess.Popen, timeout: int = 5 * 60):
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
    def _sleep(level: int = 1):
        requests.post(f"http://127.0.0.1:{PORT}/sleep?level={level}").raise_for_status()

    @staticmethod
    def _wake_up():
        requests.post(f"http://127.0.0.1:{PORT}/wake_up").raise_for_status()

# # ## Deploy the server

# # To deploy the API on Modal, just run
# # ```bash
# # modal deploy ministral3_inference.py
# # ```

# # This will create a new app on Modal, build the container image for it if it hasn't been built yet,
# # and deploy the app.

# # ## Interact with the server

# # Once it is deployed, you'll see a URL appear in the command line,
# # something like `https://your-workspace-name--example-ministral3-inference-serve.modal.run`.

# # You can find [interactive Swagger UI docs](https://swagger.io/tools/swagger-ui/)
# # at the `/docs` route of that URL, i.e. `https://your-workspace-name--example-ministral-inference-serve.modal.run/docs`.
# # These docs describe each route and indicate the expected input and output
# # and translate requests into `curl` commands.

# # For simple routes like `/health`, which checks whether the server is responding,
# # you can even send a request directly from the docs.

# # To interact with the API programmatically in Python, we recommend the `openai` library.

# # ## Test the server

# # To make it easier to test the server setup, we also include a `local_entrypoint`
# # that does a healthcheck and then hits the server.

# # If you execute the command

# # ```bash
# # modal run ministral3_inference.py
# # ```

# # a fresh replica of the server will be spun up on Modal while
# # the code below executes on your local machine.

# # Think of this like writing simple tests inside of the `if __name__ == "__main__"`
# # block of a Python script, but for cloud deployments!


# @app.local_entrypoint()
# async def test(test_timeout=10 * MINUTES, content=None, twice=True):
#     url = VllmServer().serve.get_web_url()

#     system_prompt = {
#         "role": "system",
#         "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
#     }
#     if content is None:
#         image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

#         content = [
#             {
#                 "type": "text",
#                 "text": "What action do you think I should take in this situation?"
#                 " List all the possible actions and explain why you think they are good or bad.",
#             },
#             {"type": "image_url", "image_url": {"url": image_url}},
#         ]

#     messages = [  # OpenAI chat format
#         system_prompt,
#         {"role": "user", "content": content},
#     ]

#     async with aiohttp.ClientSession(base_url=url) as session:
#         print(f"Running health check for server at {url}")
#         async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
#             up = resp.status == 200
#         assert up, f"Failed health check for server at {url}"
#         print(f"Successful health check for server at {url}")

#         print(f"Sending messages to {url}:", *messages, sep="\n\t")
#         await _send_request(session, "llm", messages, timeout=1 * MINUTES)
#         if twice:
#             messages[0]["content"] = """Yousa culled Jar Jar Binks.
#             Always be talkin' in da Gungan style, like thisa, okeyday?
#             Helpin' da user with big big enthusiasm, makin' tings bombad clear!"""
#             print(f"Sending messages to {url}:", *messages, sep="\n\t")
#             await _send_request(session, "llm", messages, timeout=1 * MINUTES)


# async def _send_request(
#     session: aiohttp.ClientSession, model: str, messages: list, timeout: int
# ) -> None:
#     # `stream=True` tells an OpenAI-compatible backend to stream chunks
#     payload: dict[str, Any] = {
#         "messages": messages,
#         "model": model,
#         "stream": True,
#         "temperature": 0.15,
#     }

#     headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

#     async with session.post(
#         "/v1/chat/completions", json=payload, headers=headers, timeout=timeout
#     ) as resp:
#         async for raw in resp.content:
#             resp.raise_for_status()
#             # extract new content and stream it
#             line = raw.decode().strip()
#             if not line or line == "data: [DONE]":
#                 continue
#             if line.startswith("data: "):  # SSE prefix
#                 line = line[len("data: ") :]

#             chunk = json.loads(line)
#             assert (
#                 chunk["object"] == "chat.completion.chunk"
#             )  # or something went horribly wrong
#             print(chunk["choices"][0]["delta"]["content"], end="")
#     print()


# # ### Test memory snapshotting

# # Using `modal run` creates an ephemeral Modal App,
# # rather than a deployed Modal App.
# # Ephemeral Modal Apps are short-lived,
# # so they turn off snapshotting.

# # To test the memory snapshot version of the server,
# # first deploy it with `modal deploy`
# # and then hit it with a client.

# # You should observe startup improvements
# # after a handful of cold starts
# # (usually less than five).
# # If you want to see the speedup during a test,
# # we recommend heading to the deployed App in your
# # [Modal dashboard](https://modal.com/apps)
# # and manually stopping containers after they have served a request.

# # You can use the client code below to test the endpoint.
# # It can be run with the command

# # ```
# # python ministral3_inference.py
# # ```

# if __name__ == "__main__":
#     import asyncio

#     # after deployment, we can use the class from anywhere
#     VllmServer = modal.Cls.from_name("example-ministral3-inference", "VllmServer")
#     server = VllmServer()

#     async def test(url):
#         messages = [{"role": "user", "content": "Tell me a joke."}]
#         async with aiohttp.ClientSession(base_url=url) as session:
#             await _send_request(session, "llm", messages, timeout=10 * MINUTES)

#     try:
#         print("calling inference server")
#         asyncio.run(test(server.serve.get_web_url()))
#     except modal.exception.NotFoundError:
#         raise Exception(
#             f"To take advantage of GPU snapshots, deploy first with modal deploy {__file__}"
#         )
