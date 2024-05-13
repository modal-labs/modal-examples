import json
import pathlib
import subprocess

import modal

comfyui_commit_sha = "0fecfd2b1a2794b77277c7e256c84de54a63d860"

comfyui_workflow_data_path = pathlib.Path(__file__).parent / "workflow-examples"

base_models = json.loads(
    (
        pathlib.Path(comfyui_workflow_data_path) / "example-model.json"
    ).read_text()
)

comfyui_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add --fetch origin https://github.com/comfyanonymous/ComfyUI",
        f"cd /root && git checkout {comfyui_commit_sha}",
        "cd /root && pip install xformers!=0.0.18 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "httpx",
        "tqdm",
        "websocket-client",
    )
)

app = modal.App(
    name="example-comfyui-api-quickstart",
)

with comfyui_image.imports():
    from .helpers import download_to_comfyui


@app.cls(
    allow_concurrent_inputs=100,
    concurrency_limit=1,
    gpu="any",
    image=comfyui_image,
    timeout=1800,
    container_idle_timeout=300,
)
class ComfyUI:
    def __init__(self, models: list[dict] = []):
        self.models = models

    @modal.build()
    def download_base_models(self):
        for model in base_models:
            download_to_comfyui(model["url"], model["path"])

    def _run_comfyui_server(self):
        # NOTE: use volume as cache?
        for model in self.models:
            download_to_comfyui(model["url"], model["path"])
        cmd = "python main.py --dont-print-server --listen --port 8188"
        subprocess.Popen(cmd, shell=True)

    @modal.enter()
    def prepare_comfyui(self):
        self._run_comfyui_server()

    # ComfyUI classic for development / debugging
    @modal.web_server(8188, startup_timeout=30)
    def ui(self):
        self._run_comfyui_server()
