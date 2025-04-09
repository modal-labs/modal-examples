# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/comfyui/was_node_suite/was_node_example.py"]
# ---

import subprocess

import modal

image = (
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("comfy-cli==1.2.7")  # install comfy-cli
    .run_commands(  # use comfy-cli to install the ComfyUI repo and its dependencies
        "comfy --skip-prompt install --nvidia"
    )
    .run_commands(  # install default stable diffusion model for example purposes
        "comfy --skip-prompt model download --url https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors --relative-path models/checkpoints"
    )
    .run_commands(  # download the WAS Node Suite custom node pack
        "comfy node install was-node-suite-comfyui"
    )
)

app = modal.App(name="example-was-node", image=image)


# Run ComfyUI as an interactive web server
@app.function(
    max_containers=1,
    scaledown_window=30,
    timeout=1800,
    gpu="A10G",
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)
