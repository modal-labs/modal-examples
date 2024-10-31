import subprocess

import modal
from comfyui.comfy_base_image import image

image = image.run_commands(  # install default stable diffusion model for example purposes
    "comfy --skip-prompt model download --url https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors --relative-path models/checkpoints"
).run_commands(  # download the WAS Node Suite custom node pack
    "comfy node install was-node-suite-comfyui"
)

app = modal.App(name="example-was-node", image=image)


# Run ComfyUI as an interactive web server
@app.function(
    allow_concurrent_inputs=10,
    concurrency_limit=1,
    container_idle_timeout=30,
    timeout=1800,
    gpu="A10G",
)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
