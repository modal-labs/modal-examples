# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/comfyui/comfyapp.py"]
# deploy: true
# ---
#
# # Run ComfyUI interactively and as an API
#
# [ComfyUI](https://github.com/comfyanonymous/ComfyUI) is a no-code Stable Diffusion GUI that allows you to design and execute advanced image generation pipelines.
#
# ![example comfyui image](./comfyui.png)
#
# In this example, we show you how to
#
# 1. run ComfyUI interactively to develop workflows
#
# 2. serve a ComfyUI workflow as an API
#
# Combining the UI and the API in a single app makes it easy to iterate on your workflow even after deployment.
# Simply head to the interactive UI, make your changes, export the JSON, and redeploy the app.
#
# ## Quickstart
#
# This example serves the [ComfyUI inpainting example workflow](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/),
# which "fills in" part of an input image based on a prompt.
# For the prompt `"Spider-Man visits Yosemite, rendered by Blender, trending on artstation"`
# on [this input image](https://raw.githubusercontent.com/comfyanonymous/ComfyUI_examples/master/inpaint/yosemite_inpaint_example.png),
# we got this output:
#
# ![example comfyui image](./comfyui_gen_image.jpg)
#
# 1. Stand up the ComfyUI server in development mode:
# ```bash
# modal serve 06_gpu_and_ml/comfyui/comfyapp.py
# ```
#
# 2. In another terminal, run inference:
# ```bash
# python 06_gpu_and_ml/comfyui/comfyclient.py --dev --modal-workspace your-modal-workspace --prompt "your prompt here"
# ```
# You can find your Modal workspace name by running `modal profile current`.
#
# The first inference will take a bit longer because the server will need to boot up (~20-30s).
# Successive inference calls while the server is up should take a few seconds or less.
#
# ## Setup
#
# First, we define the environment we need to run ComfyUI using [comfy-cli](https://github.com/Comfy-Org/comfy-cli). This handy tool manages the installation of ComfyUI, its dependencies, models, and custom nodes.


import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("comfy-cli==1.0.33")  # install comfy-cli
    .run_commands(  # use comfy-cli to install the ComfyUI repo and its dependencies
        "comfy --skip-prompt install --nvidia",
    )
    .run_commands(  # download the inpainting model
        "comfy --skip-prompt model download --url https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/512-inpainting-ema.safetensors --relative-path models/checkpoints"
    )
    .run_commands(  # download a custom node
        "comfy node install image-resize-comfyui"
    )
    # can layer additional models and custom node downloads as needed
)

app = modal.App(name="example-comfyui", image=image)


# ## Running ComfyUI interactively and as an API on Modal
#
# To run ComfyUI interactively, simply wrap the `comfy launch` command in a Modal Function and serve it as a web server.
@app.function(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=30,
    timeout=1800,
    gpu="any",
)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)


# Remember to **close your UI tab** when you are done developing to avoid accidental charges to your account.
# This will close the connection with the container serving ComfyUI, which will spin down based on your `container_idle_timeout` setting.
#
# To run an existing workflow as an API, we use Modal's class syntax to run our customized ComfyUI environment and workflow on Modal.
#
# Here's the basic breakdown of how we do it:
# 1. We stand up a "headless" ComfyUI server in the background when the app starts.
# 2. We define an `infer` method that takes in a workflow path and runs the workflow on the ComfyUI server.
# 3. We stand up an `api` with `web_endpoint`, so that we can run our workflows as a service.
#
# For more on how to run web services on Modal, check out [this guide](https://modal.com/docs/guide/webhooks).
@app.cls(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=300,
    gpu="any",
    mounts=[
        modal.Mount.from_local_file(
            Path(__file__).parent / "workflow_api.json",
            "/root/workflow_api.json",
        ),
        # mount input images
        modal.Mount.from_local_file(
            Path(__file__).parent / "yosemite_inpaint_example.png",
            "/root/comfy/ComfyUI/input/yosemite_inpaint_example.png",
        ),
    ],
)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        cmd = "comfy launch --background"
        subprocess.run(cmd, shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        # runs the comfy run --workflow command as a subprocess
        cmd = f"comfy run --workflow {workflow_path} --wait"
        subprocess.run(cmd, shell=True, check=True)

        # completed workflows write output images to this directory
        output_dir = "/root/comfy/ComfyUI/output"
        # looks up the name of the output image file based on the workflow
        workflow = json.loads(Path(workflow_path).read_text())
        file_prefix = [
            node.get("inputs")
            for node in workflow.values()
            if node.get("class_type") == "SaveImage"
        ][0]["filename_prefix"]

        # returns the image as bytes
        for f in Path(output_dir).iterdir():
            if f.name.startswith(file_prefix):
                return f.read_bytes()

    @modal.web_endpoint(method="POST")
    def api(self, item: Dict):
        from fastapi import Response

        workflow_data = json.loads(
            (Path(__file__).parent / "workflow_api.json").read_text()
        )

        # insert the prompt
        workflow_data["3"]["inputs"]["text"] = item["prompt"]

        # give the output image a unique id per client request
        client_id = uuid.uuid4().hex
        workflow_data["11"]["inputs"]["filename_prefix"] = client_id

        # save this updated workflow to a new file
        new_workflow_file = f"{client_id}.json"
        json.dump(workflow_data, Path(new_workflow_file).open("w"))

        # run inference on the currently running container
        img_bytes = self.infer.local(new_workflow_file)

        return Response(img_bytes, media_type="image/jpeg")


# ### The workflow for developing workflows
#
# When you run this script with `modal deploy 06_gpu_and_ml/comfyui/comfyapp.py`, you'll see a link that includes `ui`.
# Head there to interactively develop your ComfyUI workflow. All of your models and custom nodes specified in the image build step will be loaded in.
#
#
# To serve the workflow after you've developed it, first export it as "API Format" JSON:
# 1. Click the gear icon in the top-right corner of the menu
# 2. Select "Enable Dev mode Options"
# 3. Go back to the menu and select "Save (API Format)"
#
# Save the exported JSON to the `workflow_api.json` file in this directory.
#
# Then, redeploy the app with this new workflow by running `modal deploy 06_gpu_and_ml/comfyui/comfyapp.py` again.
#
# ## Further optimizations
#
# - If you're noticing long startup times for the ComfyUI server (e.g. >30s), this is likely due to too many custom nodes being loaded in. Consider breaking out your deployments into one App per unique combination of models and custom nodes.
# - To reduce image build time, you can write custom code to cache previous model and custom node downloads into a Modal [Volume](https://modal.com/docs/guide/volumes) to avoid full downloads on image rebuilds. (see [gist](https://gist.github.com/kning/bb5f076e831266d00e134fcb3a13ed88)).
# - For those who prefer to run a ComfyUI workflow directly as a Python script, see [this blog post](https://modal.com/blog/comfyui-prototype-to-production).
