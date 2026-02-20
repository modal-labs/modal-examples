import os

import modal

MODEL_NAME = os.environ.get("MODEL_NAME", "LiquidAI/LFM2-8B-A1B")
print(f"Running deployment script for model: {MODEL_NAME}")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.2",
        "huggingface-hub==0.36.0",
        "flashinfer-python==0.5.2",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_USE_FUSED_MOE_GROUPED_TOPK": "0",
            "MODEL_NAME": MODEL_NAME,
            "TORCH_CPP_LOG_LEVEL": "FATAL",
            "VLLM_SERVER_DEV_MODE": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("lfm-vllm-pypi-inference")

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000

with vllm_image.imports():
    import subprocess
    import time

    import requests


@app.cls(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=5 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=32)
class LfmVllmInference:
    @modal.enter(snap=True)
    def serve(self):
        print(f"Deploying model: {MODEL_NAME}")
        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            MODEL_NAME,
            f"--served-model-name {MODEL_NAME}",
            "--host 0.0.0.0",
            f"--port {str(VLLM_PORT)}",
            # extra arguments
            "--dtype bfloat16",
            "--gpu-memory-utilization 0.6",
            "--max-model-len 32768",
            "--max-num-seqs 600",
            "--enable-sleep-mode",
        ]

        cmd += ["--tensor-parallel-size", str(N_GPU)]

        print(cmd)

        subprocess.Popen(" ".join(cmd), shell=True)

        self.wait_for_server()
        self.sleep_model()

    @modal.enter(snap=False)
    def restore(self):
        self.wake_model()
        self.warmup_model()

    def wake_model(self):
        try:
            print("Waking up the model...")
            url = f"http://localhost:{VLLM_PORT}/wake_up"
            response = requests.post(url, timeout=60)
            print(f"Model is now awake. Response: {response.status_code}")
        except Exception as e:
            print(f"Failed to wake up the model: {e}")

    def sleep_model(self):
        try:
            print("Putting server into sleep mode...")
            url = f"http://localhost:{VLLM_PORT}/sleep"
            params = {"level": 1}
            response = requests.post(url, params=params, timeout=60)
            print(f"Server is now in sleep mode. Response: {response.status_code}")
        except Exception as e:
            print(f"Failed to put server into sleep mode: {e}")

    def wait_for_server(self, timeout=60 * 20, check_interval_seconds=5):
        """Wait for the VLLM server to be ready"""
        start_time = time.time()
        count = 0
        while time.time() - start_time < timeout:
            try:
                if self.healthcheck():
                    print("VLLM server is ready!")
                    return
                count += 1
            except Exception:
                time.sleep(check_interval_seconds)
        raise TimeoutError("VLLM server failed to start within timeout period")

    def healthcheck(self):
        """
        Perform a healthcheck on the VLLM server
        Returns True if healthy, False otherwise
        """
        try:
            response = requests.get(f"http://localhost:{VLLM_PORT}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def warmup_model(self, num_requests=4, timeout=60):
        try:
            print("Warming up the model...")
            url = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the melting temperature of silver?",
                    }
                ],
                "max_tokens": 256,
                "temperature": 0,
            }

            for i in range(num_requests):
                print(f"Warmup request {i + 1}/4...")
                response = requests.post(url, json=payload, timeout=timeout)
                print(
                    f"Warmup request {i + 1} completed with status: {response.status_code}"
                )

            print("Warmup complete!")
        except Exception as e:
            print(f"Warmup failed: {e}")

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
    def webserver(self):
        pass
