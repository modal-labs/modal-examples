# ---
# deploy: true
# cmd: ["modal", "serve", "06_gpu_and_ml/comfyui/comfyapp.py"]
# ---
#
# # Run Flux on ComfyUI interactively and as an API
#
# [ComfyUI](https://github.com/comfyanonymous/ComfyUI) is an open-source diffusion model platform with a graph/nodes interface that allows you to design and execute advanced image generation pipelines.

#
# In this example, we show you how to
#
# 1. run the [Flux](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux) diffusion model on ComfyUI interactively to develop workflows
#
# 2. serve a Flux ComfyUI workflow as an API
#
# ## Quickstart
#
# This example runs `workflow_api.json` in this directory, which is an adapation of [this simple FLUX.1-schnell workflow](https://openart.ai/workflows/reverentelusarca/flux-simple-workflow-schnell/40OkdaB23J2TMTXHmxxu) with an Image Resize custom node added at the end.
#
# For the prompt `"Surreal dreamscape with floating islands, upside-down waterfalls, and impossible geometric structures, all bathed in a soft, ethereal light"`
# we got this output:
#
# ![example comfyui image](./flux_gen_image.jpeg)
#
# To serve the workflow in this example as an API:
# 1. Download the Flux models to a Modal [Volume](/docs/guide/volumes):
# ```bash
# modal run 06_gpu_and_ml/comfyui/comfyapp.py::download_models
# ```
#
# 2. Stand up the ComfyUI server in development mode:
# ```bash
# modal serve 06_gpu_and_ml/comfyui/comfyapp.py
# ```
#
# 3. In another terminal, run inference:
# ```bash
# python 06_gpu_and_ml/comfyui/comfyclient.py --dev --modal-workspace $(modal profile current) --prompt "neon green sign that says Modal"
# ```
#
# The first inference will take ~1m since the container needs to launch the ComfyUI server and load Flux into memory. Successive inferences on a warm container should take a few seconds.
#
# ## Setup
#

import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict

import modal

# ### Building up the environment
#
# We start from a base image and specify all of our dependencies.
# We'll call out the interesting ones as they come up below.
#
# Note that these dependencies are not installed locally.
# They are only installed in the remote environment where our app runs.
# This happens the first time. On subsequent runs, the cached image will be reused.


image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("fastapi[standard]==0.115.4")  # install web dependencies
    .pip_install("comfy-cli==1.2.7")  # install comfy-cli
    .run_commands(  # use comfy-cli to install the ComfyUI repo and its dependencies
        "comfy --skip-prompt install --nvidia"
    )
)
# ### Downloading custom nodes
# We'll use `comfy-cli` to download custom nodes, in this case the popular WAS Node Suite pack.
image = (
    image.run_commands(  # download a custom node
        "comfy node install was-node-suite-comfyui"
    )
    # Add .run_commands(...) calls for any other custom nodes you want to download
)

# See [this post](/blog/comfyui-custom-nodes) for more on how to install custom nodes on Modal.
# ### Downloading models

# You can also use comfy-cli to download models, but for this example we'll download the Flux models directly from Hugging Face into a Modal Volume.
# Then on container start, we'll mount our models into the ComfyUI models directory.
# This allows us to avoid re-downloading the models every time you rebuild your image.

image = (
    # install huggingface_hub with hf_transfer support to speed up downloads
    image.pip_install("huggingface_hub[hf_transfer]==0.26.2")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_commands(  # needs to be empty for Volume mount to work
        "rm -rf /root/comfy/ComfyUI/models"
    )
)

# We create the app and specify the image we built above.

app = modal.App(name="example-comfyui", image=image)

#
# First we need to run a function to download the Flux models to a Modal Volume.

vol = modal.Volume.from_name("comfyui-models", create_if_missing=True)


@app.function(
    volumes={"/root/models": vol},
)
def hf_download(repo_id: str, filename: str, model_type: str):
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=f"/root/models/{model_type}",
    )


# We can kick off the model downloads in parallel using [`starmap`](/docs/reference/modal.Function#starmap).
@app.local_entrypoint()
def download_models():
    models_to_download = [
        # format is (huggingface repo_id, the model filename, comfyui models subdirectory we want to save the model in)
        (
            "black-forest-labs/FLUX.1-schnell",
            "ae.safetensors",
            "vae",
        ),
        (
            "black-forest-labs/FLUX.1-schnell",
            "flux1-schnell.safetensors",
            "unet",
        ),
        (
            "comfyanonymous/flux_text_encoders",
            "t5xxl_fp8_e4m3fn.safetensors",
            "clip",
        ),
        ("comfyanonymous/flux_text_encoders", "clip_l.safetensors", "clip"),
    ]
    list(hf_download.starmap(models_to_download))


# To run the download step, run `modal run 06_gpu_and_ml/comfyui/comfyapp.py::download_models`.
# By leveraging [hf_transfer](https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads), Modal starmap for parallelism, and Volumes, image build time drops from ~10 minutes to ~25 seconds.

# ## Running ComfyUI interactively and as an API on Modal
#
# To run ComfyUI interactively, simply wrap the `comfy launch` command in a Modal Function and serve it as a web server.


@app.function(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=30,
    timeout=1800,
    gpu="A10G",
    volumes={"/root/comfy/ComfyUI/models": vol},
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
    container_idle_timeout=300,
    gpu="A10G",
    mounts=[
        modal.Mount.from_local_file(
            Path(__file__).parent / "workflow_api.json",
            "/root/workflow_api.json",
        ),
    ],
    volumes={"/root/comfy/ComfyUI/models": vol},
)
class ComfyUI:
    @modal.enter()
    def launch_comfy_background(self):
        cmd = "comfy launch --background"
        subprocess.run(cmd, shell=True, check=True)

    @modal.method()
    def infer(self, workflow_path: str = "/root/workflow_api.json"):
        # runs the comfy run --workflow command as a subprocess
        cmd = f"comfy run --workflow {workflow_path} --wait --timeout 1200"
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
        workflow_data["6"]["inputs"]["text"] = item["prompt"]

        # give the output image a unique id per client request
        client_id = uuid.uuid4().hex
        workflow_data["9"]["inputs"]["filename_prefix"] = client_id

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
# - To decrease inference latency, you can process multiple inputs in parallel by setting `allow_concurrent_inputs=1`, which will run each input on its own container. See our [Scaling ComfyUI](https://modal.com/blog/scaling-comfyui) blog post for more details.
# - If you're noticing long startup times for the ComfyUI server (e.g. >30s), this is likely due to too many custom nodes being loaded in. Consider breaking out your deployments into one App per unique combination of models and custom nodes.
# - For those who prefer to run a ComfyUI workflow directly as a Python script, see [this blog post](https://modal.com/blog/comfyui-prototype-to-production).
