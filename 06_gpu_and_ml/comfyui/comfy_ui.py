# ---
# cmd: ["modal", "serve", "06_gpu_and_ml.comfyui.comfy_ui"]
# ---
#
# # Run a ComfyUI workflow as an API
#
# [ComfyUI](https://github.com/comfyanonymous/ComfyUI) is a no-code Stable Diffusion GUI that allows you to design and execute advanced image generation pipelines.
#
# In this example, we show you how to
#
# 1) Run ComfyUI interactively
#
# 2) Optimize performance with [@enter](/docs/guide/lifecycle-functions#enter)
#
# 3) Run a ComfyUI workflow JSON via API
#
# The primary goal of this example is to shows users an easy way to deploy an existing ComfyUI workflow on Modal.
# This also covers some more advanced concepts on performance optimization, and so we assume you have some familiarity with ComfyUI already.
#
# An alternative approach is to port your ComfyUI workflow from JSON into Python, which you can check out [in this blog post](/blog/comfyui-prototype-to-production).
# The Python approach reduces latency by skipping the server standup step entirely, but requires more effort to migrate to from JSON.
#
# ## Quickstart
# 1) Run `cd 06_gpu_and_ml`
#
# 2) Run `modal serve comfyui.comfy_ui` to stand up the ComfyUI server.
# This example serves the [ComfyUI inpainting example workflow](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/) behind an API.
# Inpainting is the process of filling in an image with another generated image.
#
# 3) Run inference with a text prompt: `python -m comfyui.infer --prompt "white heron"`. This creates the following image:
# ![example comfyui image](./comfyui_gen_image.jpg)
#
# Try running inference again with a different prompt e.g. `python -m comfyui.infer.py --prompt "white tiger"`.
# Notice how successive inference calls are much faster. In our tests, inference calls drop from 30s to 3s due to our optimized performance design.
#
# Now we'll dive into the step-by-step process of how to run ComfyUI both interactively and as an API, as well as how we're able to leverage Modal classes to run arbitrary workflows with minimal cold starts.
# ## Run ComfyUI interactively
# First, we define the ComfyUI image.

import json
import pathlib
import subprocess

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
    .pip_install(
        "httpx",
        "tqdm",
        "websocket-client",
    )
)

app = modal.App(
    name="example-comfyui",
)

# You can define custom checkpoints, plugins, and more in the `workflow-examples/base-model.json` in this directory.

comfyui_workflow_data_path = pathlib.Path(__file__).parent / "workflow-examples"
base_models = json.loads(
    (pathlib.Path(comfyui_workflow_data_path) / "base-model.json").read_text()
)

# Specific workflows (like our inpainting example) have their own folder containing the workflow JSON as well as that workflow's corresponding `model.json` which specifies the custom checkpoints/plugins used in the workflow.
# These get loaded once at container start time and not build time; we'll go into more detail on how that works in the next section.
#
# We move a lot of ComfyUI-specific code to `helpers.py`.
# This includes functions like downloading checkpoints/plugins to the right directory on the ComfyUI server.
with comfyui_image.imports():
    from .helpers import (
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
)
class ComfyUI:
    def __init__(self, models: list[dict] = []):
        self.models = models

    @modal.build()
    def download_base_models(self):
        for model in base_models:
            download_to_comfyui(model["url"], model["path"])

    def _run_comfyui_server(self, port=8188):
        for model in self.models:
            download_to_comfyui(model["url"], model["path"])
        cmd = f"python main.py --dont-print-server --listen --port {port}"
        subprocess.Popen(cmd, shell=True)

    @modal.web_server(8188, startup_timeout=30)
    def ui(self):
        self._run_comfyui_server()

    # When you run `modal serve comfyui.comfy_ui`, you'll see a `ComfyUI.ui` link to interactively develop your ComfyUI workflow that has the custom checkpoints/plugins loaded in.
    #
    # Notice the `__init__` constructor.
    # This allows us to leverage a special Modal pattern called [parameterized functions](/docs/guide/lifecycle-functions#parametrized-functions) that will allow us to support arbitrary workflows and custom checkpoints/plugins in an optimized way.
    #
    # ## Optimize performance with `@enter`
    #
    # By setting a `models` argument for the class, we can dynamically download arbitrary models at runtime on top of the base objects that were downloaded at `@build` time.
    # We can use the `@enter` function to optimize inference time by downloading custom models and standing up the ComfyUI server exactly once at container startup time.
    @modal.enter()
    def prepare_comfyui(self):
        self._run_comfyui_server(port=8189)

    # Lastly, we write the inference method that takes in any workflow JSON and additional arguments you may want to use to parameterize your workflow JSON (e.g. handle user-defined text prompts, input images).
    # It then runs the workflow programmatically against the running ComfyUI server and returns the images.
    @modal.method()
    def infer(self, workflow_data: dict, params: dict):
        # input images need to be downloaded to the container at this step
        download_to_comfyui(params["input_image_url"], "input")

        # insert custom text prompt
        workflow_data["3"]["inputs"]["text"] = params["text_prompt"]
        ws = connect_to_local_server()
        images = get_images(ws, workflow_data)
        return images


# ## Run a ComfyUI workflow JSON via API
# Now we have our ComfyUI class fully defined, we can stand up a simple backend to receive requests.

web_image = modal.Image.debian_slim()

from typing import Dict


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


# To deploy this API, run `modal deploy comfyui.comfy_ui`

# ## Further optimization
# After deploying, you can also apply [keep warm](/docs/reference/modal.Function#keep_warm) to a particular `model.json` combination of checkpoints/plugins.
# This will stand up a dedicated container pool for any workflows that have the same `model.json` config.
# This can help you further minimize a harsh cold start when a workflow is run for the first time.


@app.local_entrypoint()
def apply_config():
    DeployedComfyUI = modal.Cls.lookup("example-comfyui", "ComfyUI")
    models = json.loads(
        pathlib.Path(
            comfyui_workflow_data_path / "inpainting" / "model.json"
        ).read_text()
    )
    DeployedComfyUI(models).infer.keep_warm(1)


# Some other things to try:
#
# * Cache downloaded checkpoints/plugins to a [Volume](https://modal.com/docs/guide/volumes) in the `@enter` step and load from there in successive cold starts.
# * Move common checkpoints/plugins to the `@build` step instead of `@enter`.
