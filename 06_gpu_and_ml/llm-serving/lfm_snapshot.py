import os
import subprocess
import time

import aiohttp
import modal

MINUTES = 60

MODEL_NAME = os.environ.get("MODEL_NAME", "LiquidAI/LFM2-8B-A1B")
print(f"Running deployment script for model: {MODEL_NAME}")

vllm_image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.15.1")
    .entrypoint([])
    .run_commands(
        "ln -s $(which python3) /usr/bin/python"
    )
    .pip_install("transformers==5.1.0")
    .env(
        {
            "HF_HUB_CACHE": "/root/.cache/huggingface",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "MODEL_NAME": MODEL_NAME
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("examples-lfm-snapshot")

N_GPU = 1
VLLM_PORT = 8000
TARGET_INPUTS = 10
MAX_INPUTS = 100

with vllm_image.imports():
    import requests


def wait_ready(process: subprocess.Popen, timeout: int = 15 * MINUTES):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:{VLLM_PORT}/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(5)
    raise TimeoutError(f"vLLM server not ready within {timeout} seconds")


def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def warmup():
    payload = {
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }
    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            json=payload,
            timeout=60,
        ).raise_for_status()


@app.cls(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=5 * MINUTES,
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret-liquid")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(target_inputs=TARGET_INPUTS, max_inputs=MAX_INPUTS)
class LfmVllmInference:
    @modal.enter(snap=True)
    def startup(self):
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "--served-model-name",
            "llm",
            "--host",
            "0.0.0.0",
            "--port",
            f"{VLLM_PORT}",
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.8",
            "--max-num-seqs",
            f"{MAX_INPUTS}",
            "--max-cudagraph-capture-size",
            f"{MAX_INPUTS}",
            "--enable-sleep-mode",
        ]

        print(*cmd)
        self.process = subprocess.Popen(cmd)
        wait_ready(self.process)
        warmup()

    @modal.enter(snap=False)
    def restore(self):
        pass

    @modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
    def serve(self):
        pass

    @modal.exit()
    def stop(self):
        self.process.terminate()


@app.local_entrypoint()
async def test():
    url = await LfmVllmInference().serve.get_web_url.aio()
    print(f"Server URL: {url}")

    messages = [
        {"role": "user", "content": "Count to 1000, slowly."},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        async with session.get(
            "/health", timeout=aiohttp.ClientTimeout(total=15 * MINUTES)
        ) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
            print("Health check passed")

        while True:
            payload = {
                "model": "llm",
                "messages": messages,
                "max_tokens": 7 * 1024,
                "temperature": 0,
            }
            async with session.post(
                "/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                print(f"Response: {content}")
