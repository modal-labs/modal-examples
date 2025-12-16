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
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TMS_INIT_ENABLE_CPU_BACKUP": "1",
            "TORCHINDUCTOR_CACHE_DIR": "/root/.cache/torch/",
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
            "--enable-metrics",
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
