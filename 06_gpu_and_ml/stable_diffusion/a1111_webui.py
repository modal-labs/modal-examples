# ---
# lambda-test: false
# ---
# # Stable Diffusion (A1111)
#
# This example runs the popular [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
# project on Modal, without modification. We start a Modal container with an A10G GPU, run the server as a
# subprocess, and forward the port using a [tunnel](/docs/guide/tunnels).

import socket
import subprocess
import time
import webbrowser

from modal import Image, Queue, Stub, forward

stub = Stub("example-a1111-webui")
stub.urls = Queue.new()  # TODO: FunctionCall.get() doesn't support generators.


def wait_for_port(port: int):
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=5.0):
                break
        except OSError:
            time.sleep(0.1)


# The following function starts the web UI container. Notice that it requires a few steps to
# install dependencies, since `stable-diffusion-webui` doesn't come with a prepackaged script
# to do this. (It usually installs dependencies on first launch.)
#
# After defining the custom container image, we start the server with `accelerate launch`. This
# function is also where you would configure hardware resources, CPU/memory, and timeouts.
#
# If you want to run it with an A100 GPU, just change `gpu="a10g"` to `gpu="a100"`.


@stub.function(
    image=Image.debian_slim()
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
    ),
    gpu="a10g",
    cpu=2,
    memory=1024,
    timeout=3600,
)
def start_web_ui():
    START_COMMAND = r"""
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
        --listen \
        --port 8000
"""
    with forward(8000) as tunnel:
        p = subprocess.Popen(START_COMMAND, shell=True)
        wait_for_port(8000)
        print("[MODAL] ==> Accepting connections at", tunnel.url)
        stub.urls.put(tunnel.url)
        p.wait(3600)


# The first run may take a few minutes to build the image. When the container starts, it will open
# the page in your browser.


@stub.local_entrypoint()
def main(no_browser: bool = False):
    start_web_ui.spawn()
    url = stub.urls.get()
    if not no_browser:
        webbrowser.open(url)
    while True:  # TODO: FunctionCall.get() doesn't support generators.
        time.sleep(1)
