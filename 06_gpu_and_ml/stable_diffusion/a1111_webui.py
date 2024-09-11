# ---
# lambda-test: false
# ---
# # Stable Diffusion (A1111)
#
# This example runs the popular [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
# project on Modal, without modification. We just port the environment setup to a Modal container image
# and wrap the launch script with a `@web_server` decorator, and we're ready to go.
#
# You can run a temporary A1111 server with `modal serve a1111_webui.py` or deploy it permanently with `modal deploy a1111_webui.py`.

import subprocess

import modal

PORT = 8000

# First, we define the image A1111 will run in.
# This takes a few steps because A1111 usually install its dependencies on launch via a script.
# The process may take a few minutes the first time, but subsequent image builds should only take a few seconds.

a1111_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "git",
        "libgl1",
        "libglib2.0-0",
        "google-perftools",  # For tcmalloc
    )
    .env({"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"})
    .run_commands(
        "git clone --depth 1 --branch v1.7.0 https://github.com/AUTOMATIC1111/stable-diffusion-webui /webui",
        "python -m venv /webui/venv",
        "cd /webui && . venv/bin/activate && "
        + "python -c 'from modules import launch_utils; launch_utils.prepare_environment()' --xformers",
        gpu="a10g",
    )
    .run_commands(
        "cd /webui && . venv/bin/activate && "
        + "python -c 'from modules import shared_init, initialize; shared_init.initialize(); initialize.initialize()'",
        gpu="a10g",
    )
)

app = modal.App("example-a1111-webui", image=a1111_image)

# After defining the custom container image, we start the server with `accelerate launch`. This
# function is also where you would configure hardware resources, CPU/memory, and timeouts.
#
# If you want to run it with an A100 or H100 GPU, just change `gpu="a10g"` to `gpu="a100"` or `gpu="h100"`.
#
# Startup of the web server should finish in under one to three minutes.


@app.function(
    gpu="a10g",
    cpu=2,
    memory=1024,
    timeout=3600,
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    # Keep at least one instance of the server running.
    keep_warm=1,
)
@modal.web_server(port=PORT, startup_timeout=180)
def run():
    START_COMMAND = f"""
cd /webui && \
. venv/bin/activate && \
accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=fp16 \
    --dynamo_backend=inductor \
    --num_cpu_threads_per_process=6 \
    /webui/launch.py \
        --skip-prepare-environment \
        --no-gradio-queue \
        --listen \
        --port {PORT}
"""
    subprocess.Popen(START_COMMAND, shell=True)
