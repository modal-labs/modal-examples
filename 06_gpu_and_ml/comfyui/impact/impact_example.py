# ---
# cmd: ["modal", "serve", "06_gpu_and_ml/comfyui/impact/impact_example.py"]
# ---

import subprocess

import modal

image = (
    modal.Image.debian_slim(  # start from basic Linux with Python
        python_version="3.11"
    )
    .apt_install("git")  # install git to clone ComfyUI
    .pip_install("comfy-cli==1.2.7")  # install comfy-cli
    .run_commands(  # download the Impact pack
        "comfy node install ComfyUI-Impact-Pack"
    )
    .pip_install("ultralytics")  # object detection models
    .apt_install(  # opengl dependencies
        "libgl1-mesa-glx", "libglib2.0-0"
    )
    .run_commands(  # install civit ai model (you need to create a Modal Secret with your Civit AI token)
        "comfy --skip-prompt model download --url https://civitai.com/api/download/models/146134 --relative-path models/checkpoints --set-civitai-api-token $CIVIT_AI_TOKEN",
        secrets=[modal.Secret.from_name("civitai-token")],
    )
)

app = modal.App(name="example-impact", image=image)


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
