import json
import pathlib
import subprocess
from typing import Dict

import modal

comfyui_commit_sha = "0fecfd2b1a2794b77277c7e256c84de54a63d860"

comfyui_workflow_data_path = pathlib.Path(__file__).parent / "workflow-examples"

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
    from .helpers import (
        connect_to_local_server,
        download_to_comfyui,
        get_images,
    )


@app.cls(
    allow_concurrent_inputs=100,
    concurrency_limit=1,
    gpu="any",
    image=comfyui_image,
    timeout=1800,
    container_idle_timeout=300,
)
class ComfyUI:
    def __init__(self, models: list[dict]):
        self.models = models

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

    @modal.method()
    def infer(self, workflow_data: dict, params: dict):
        # TODO: server times out randomly? is it comfyui or modal? => think it might have to do with mounted file updates that invalidate the container
        # input images need to be downloaded to the container at this step
        download_to_comfyui(params["input_image_url"], "input")

        # insert custom text prompt
        workflow_data["3"]["inputs"]["text"] = params["text_prompt"]
        ws = connect_to_local_server()
        images = get_images(ws, workflow_data)
        return images


web_image = modal.Image.debian_slim()


@app.function(image=web_image, container_idle_timeout=300)
@modal.web_endpoint(method="POST")
def backend(item: Dict):
    from fastapi import Response

    workflow = json.loads(item["workflow_data"])
    models = json.loads(item["models"])
    params = {
        "text_prompt": item["text_prompt"],
        "input_image_url": item["input_image_url"],
    }
    images = ComfyUI(models).infer.remote(workflow, params)
    return Response(content=images[0], media_type="image/jpeg")


@app.local_entrypoint()
def apply_config():
    models = json.loads(
        pathlib.Path(
            comfyui_workflow_data_path / "inpainting" / "model.json"
        ).read_text()
    )
    # TODO: this doesn't seem to work
    ComfyUI(models).infer.keep_warm(1)
