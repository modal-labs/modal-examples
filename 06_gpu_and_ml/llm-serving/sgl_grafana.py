import asyncio
import os
import subprocess
import time
from pathlib import Path

import aiohttp
import modal

here = Path(__file__).parent

app = modal.App("sglang-grafana")

MODEL_ID = "Qwen/Qwen3-8B-FP8"
HOSTNAME = "0.0.0.0"
PUBLIC_PORT = 8000
MINUTES = 60

volume = modal.Volume.from_name("sglang-grafana-volume", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .pip_install(
        "huggingface_hub==0.34.0",
        "hf-transfer==0.1.9",
        "grpclib==0.4.8",
        "requests==2.32.4",
        "fastapi[standard]==0.116.1",
    )
    .pip_install(
        "accelerate==1.10.0",
        "transformers==4.55.4",
    )
    .run_commands(
        "mkdir -p /etc/apt/keyrings/",
        "wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | tee /etc/apt/keyrings/grafana.gpg > /dev/null",
        "echo 'deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main' | tee /etc/apt/sources.list.d/grafana.list",
    )
    .apt_install("alloy")
    .env({
        "HF_HOME": str(MODELS_PATH),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .dockerfile_commands(
        [
            "RUN echo '{%- for message in messages %}{{- message.content }}"
            "{%- endfor %}' > /home/no-system-prompt.jinja",
            "ENTRYPOINT []",
        ],
    )
)


def wait_for_port(process, port: int):
    import socket
    while True:
        try:
            with socket.create_connection((HOSTNAME, port), timeout=1):
                break
        except (ConnectionRefusedError, OSError):
            if process.poll() is not None:
                raise Exception(f"Process {process.pid} exited with code {process.returncode}")
@app.cls(
    image=sglang_image,
    gpu="H100",
    volumes={VOLUME_PATH: volume},
    max_containers=2,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("grafana-secret")
    ],
)
@modal.concurrent(max_inputs=8)
class SGLang():

    @modal.enter()
    def enter(self):
        import requests
        import torch

        print("writing alloy config")
        container_alloy_config = (
            get_alloy_config_template()
            .replace("__USERNAME_REPLACE_ME__", os.getenv("GRAFANA_USERNAME"))
            .replace("__API_KEY_REPLACE_ME__", os.getenv("GRAFANA_API_KEY"))
            .replace("__CONTAINER_ID_REPLACE_ME__", os.getenv("MODAL_TASK_ID", "unknown"))
        )
        Path("/etc/alloy/config.alloy").write_text(container_alloy_config)


        server_config = {
            "model-path": MODEL_ID,
            "host": HOSTNAME,
            "port": PUBLIC_PORT,
            "mem-fraction": 0.7,
            "cuda-graph-max-bs": 8,
            "enable-metrics": "",
            "tp": torch.cuda.device_count(),
        }

        args_string = " ".join([f"--{k} {v}" for k, v in server_config.items()])
        cmd = f"python -m sglang.launch_server {args_string}"
        print(f"starting server via: {cmd}")

        self.serve_process = subprocess.Popen(cmd, shell=True)
        wait_for_port(self.serve_process, PUBLIC_PORT)

        print("sending test request to warm up model")
        payload = {
            "model": MODEL_ID,
            "prompt": "Hello, world!",
            "max_tokens": 1,
        }
        while True:
            response = requests.post(
                f"http://{HOSTNAME}:{PUBLIC_PORT}/v1/completions",
                json=payload,
            )
            if response.status_code == 200:
                break
            print(f"Server not ready: {response.status_code} {response.text}")
            time.sleep(1)

        print("starting alloy process")
        self.alloy_process = subprocess.Popen(
            ["alloy", "run", "/etc/alloy/config.alloy"],
            preexec_fn=os.setsid,
        )

    @modal.web_server(PUBLIC_PORT)
    def serve(self):
        return

    @modal.exit()
    def exit(self):
        self.alloy_process.terminate()
        self.serve_process.terminate()

async def send_request(session, url, i):
    data = {
        "model": MODEL_ID,
        "prompt": f"{i % 20}. Explain quantum computing",
        "max_tokens": 256
    }
    async with session.post(f"{url}/v1/completions", json=data) as response:
        result = await response.json()
        print(f"Request {i}: {result.get('choices', [{}])[0].get('text', 'Error')}")

@app.local_entrypoint()
async def main():
    url = SGLang().serve.get_web_url()
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[send_request(session, url, i) for i in range(100)])

def get_alloy_config_template():
    return r"""
prometheus.remote_write "grafanacloud" {
  endpoint {
    url = "https://prometheus-prod-36-prod-us-west-0.grafana.net/api/prom/push"

    basic_auth {
      username = "__USERNAME_REPLACE_ME__"
      password = "__API_KEY_REPLACE_ME__"
    }
  }
  external_labels = {
    "container_id" = "__CONTAINER_ID_REPLACE_ME__",
  }
}

prometheus.scrape "llm" {
  targets         = [{"__address__" = "127.0.0.1:8000", "job" = "llm"}] // adjust to your app’s port
  scrape_interval = "10s"
  forward_to      = [prometheus.relabel.set_instance.receiver]
}

prometheus.relabel "set_instance" {
  rule {
    action       = "replace"
    target_label = "instance"
    replacement  = "__CONTAINER_ID_REPLACE_ME__"
  }
  forward_to = [prometheus.remote_write.grafanacloud.receiver]
}
"""


