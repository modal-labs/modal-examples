# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/comfyui/ip_adapter/ip_adapter_example.py"]
# ---

import subprocess

import modal

image = (  # build up a Modal Image to run ComfyUI, step by step
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("comfy-cli==1.2.7")  # install comfy-cli
    .run_commands(  # use comfy-cli to install the ComfyUI repo and its dependencies
        "comfy --skip-prompt install --nvidia"
    )
    .run_commands(  # download the WAS Node Suite custom node pack
        "comfy node install comfyui_ipadapter_plus"
    )
    .run_commands("apt install -y wget")
    .run_commands(  # the Unified Model Loader node requires these two models to be named a specific way, so we use wget instead of the usual comfy model download command
        "wget -q -O /root/comfy/ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
    )
    .run_commands(
        "wget -q -O /root/comfy/ComfyUI/models/clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors, https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors",
    )
    .run_commands(  # download the IP-Adapter model
        "comfy --skip-prompt model download --url https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors --relative-path models/ipadapter"
    )
    .run_commands(
        "comfy --skip-prompt model download --url https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors --relative-path models/checkpoints",
    )
)

app = modal.App(name="example-ip-adapter", image=image)


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
