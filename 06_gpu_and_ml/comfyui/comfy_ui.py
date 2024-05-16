# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/comfyui/comfy_ui.py"]
# ---
#
# # Run a ComfyUI workflow as an API
#
# [ComfyUI](https://github.com/comfyanonymous/ComfyUI) is a no-code Stable Diffusion GUI that allows you to design and execute advanced image generation pipelines.
#
# ![example comfyui image](./comfyui.png)
#
# In this example, we show you how to
#
# 1. Run ComfyUI interactively
#
# 2. Serve a ComfyUI workflow as an API
#
# The primary goal of this example is to shows users an easy way to deploy an existing ComfyUI workflow on Modal.
# This unified UI / API example also makes it easy to iterate on your workflow even after deployment.
# Simply fire up the interactive UI, make your changes, export the JSON, and redeploy the app.
#
# An alternative approach is to port your ComfyUI workflow from JSON into Python, which you can check out [in this blog post](/blog/comfyui-prototype-to-production).
# The Python approach further reduces inference latency by skipping the server standup step entirely, but requires more effort to migrate to from JSON.
#
# ## Quickstart
#
# 1. Run `modal serve 06_gpu_and_ml/comfyui/comfy_ui.py` to stand up the ComfyUI server.
# This example serves the [ComfyUI inpainting example workflow](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/) behind an API.
# Inpainting is the process of filling in an image with another generated image.
#
# 2. Run inference with a text prompt: `python 06_gpu_and_ml/comfyui/infer.py --prompt "white heron"`. This creates the following image:
# ![example comfyui image](./comfyui_gen_image.jpg)
#
# First inference time will take a bit longer for the ComfyUI server to boot (~30s). Successive inference calls while the server is up should take ~3s.
# ## Run ComfyUI interactively
# First, we define the ComfyUI image.

import json
import pathlib
import subprocess
from typing import Dict

import modal

comfyui_commit_sha = "0fecfd2b1a2794b77277c7e256c84de54a63d860"

comfyui_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add --fetch origin https://github.com/comfyanonymous/ComfyUI",
        f"cd /root && git checkout {comfyui_commit_sha}",
        "cd /root && pip install xformers!=0.0.18 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install("httpx", "tqdm", "websocket-client")
    .copy_local_file(
        pathlib.Path(__file__).parent / "model.json", "/root/model.json"
    )
    .copy_local_file(
        pathlib.Path(__file__).parent / "helpers.py", "/root/helpers.py"
    )
)

app = modal.App(
    name="example-comfyui",
)

# You can define custom checkpoints, plugins, and more in the `model.json` file in this directory.


# ComfyUI-specific code lives in `helpers.py`.
# This includes functions like downloading checkpoints/plugins to the right directory on the ComfyUI server.
with comfyui_image.imports():
    from helpers import (
        connect_to_local_server,
        download_to_comfyui,
        get_images,
    )


# Here we use Modal's class syntax to build the image (with our custom checkpoints/plugins).
@app.cls(
    allow_concurrent_inputs=100,
    gpu="any",
    image=comfyui_image,
    timeout=1800,
    container_idle_timeout=300,
    mounts=[
        modal.Mount.from_local_file(
            pathlib.Path(__file__).parent / "workflow_api.json",
            "/root/workflow_api.json",
        )
    ],
)
class ComfyUI:
    @modal.build()
    def download_models(self):
        models = json.loads(
            (pathlib.Path(__file__).parent / "model.json").read_text()
        )
        for m in models:
            download_to_comfyui(m["url"], m["path"])

    def _run_comfyui_server(self, port=8188):
        cmd = f"python main.py --dont-print-server --listen --port {port}"
        subprocess.Popen(cmd, shell=True)

    @modal.web_server(8188, startup_timeout=30)
    def ui(self):
        self._run_comfyui_server()

    # When you run `modal serve 06_gpu_and_ml/comfyui/comfy_ui.py`, you'll see a `ComfyUI.ui` link to interactively develop your ComfyUI workflow that has the custom checkpoints/plugins loaded in.
    #
    # To serve this workflow, first export it to API JSON format:
    # 1. Click the gear icon in the top-right corner of the menu
    # 2. Select "Enable Dev mode Options"
    # 3. Go back to the menu and select "Save (API Format)"
    #
    # Save the exported JSON to the `workflow_api.json` file in this directory.
    #
    # ## Serve a ComfyUI workflow as an API
    #
    # We use the `@enter` function to stand up a "headless" ComfyUI at container startup time.
    @modal.enter()
    def prepare_comfyui(self):
        # Runs on a different port as to not conflict with the UI instance above.
        self._run_comfyui_server(port=8189)

    # Lastly, we stand up an API web endpoint that runs the ComfyUI workflow JSON programmatically and returns the generated image.
    @modal.web_endpoint(method="POST")
    def api(self, item: Dict):
        from fastapi import Response

        # download input images to the container
        download_to_comfyui(item["input_image_url"], "input")
        workflow_data = json.loads(
            (pathlib.Path(__file__).parent / "workflow_api.json").read_text()
        )

        # insert custom text prompt
        workflow_data["3"]["inputs"]["text"] = item["prompt"]

        # send requests to local headless ComfyUI server (on port 8189)
        server_address = "127.0.0.1:8189"
        ws = connect_to_local_server(server_address)
        images = get_images(ws, workflow_data, server_address)
        return Response(content=images[0], media_type="image/jpeg")


# To deploy this API, run `modal deploy 06_gpu_and_ml/comfyui/comfy_ui.py`.

# ## Further optimizations
# There is more you can do with Modal to further improve performance of your ComfyUI API endpoint. For example:
# * Apply [keep_warm](https://modal.com/docs/guide/cold-start#maintain-a-warm-pool-with-keep_warm) to the ComfyUI class to always have a server running
# * Cache downloaded checkpoints/plugins to a [Volume](https://modal.com/docs/guide/volumes) to avoid full downloads on image rebuilds
#
# If you're interested in serving arbitrary ComfyUI workflows with arbitrary sets of custom checkpoints/plugins, please [reach out to us on Slack](https://modallabscommunity.slack.com/join/shared_invite/zt-2a4ojve51-bc89MNAk2yqOFgwqCnqicw#/shared-invite/email) and we can try to help.
