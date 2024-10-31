import subprocess

import modal
from comfyui.comfy_base_image import image

image = (  # build up a Modal Image to run ComfyUI, step by step
    image.run_commands(  # download the ComfyUI Essentials custom node pack
        "comfy node install ComfyUI_essentials"
    ).run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors --relative-path models/checkpoints"
    )
)

app = modal.App(name="example-essentials", image=image)


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
