# ---
# lambda-test: false
# ---
#
# # Run ComfyUI
#
# This example shows you how to run a ComfyUI workspace with `modal serve`.
#
# If you're unfamiliar with how ComfyUI works we recommend going through Scott Detweiler's
# [tutorials on Youtube](https://www.youtube.com/watch?v=AbB33AxrcZo).
#
# ![example comfyui workspace](./comfyui-hero.png)

import pathlib
import subprocess

import modal

# ## Define container image
#
# Fun with ComfyUI begins with pre-trained model checkpoints.
# The checkpoint downloaded below is [huggingface.co/dreamlike-art/dreamlike-photoreal-2.0](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0), but others can be used.
# The ComfyUI repository has other recommendations listed in this file:
# [notebooks/comfyui_colab.ipynb](https://github.com/comfyanonymous/ComfyUI/blob/master/notebooks/comfyui_colab.ipynb).
#
# This download function is run as the final image building step, and takes around 10 seconds to download
# the ~2.0 GiB model checkpoint.


def download_checkpoint():
    import httpx
    from tqdm import tqdm

    url = "https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/resolve/main/dreamlike-photoreal-2.0.safetensors"
    checkpoints_directory = "/root/models/checkpoints"
    local_filename = url.split("/")[-1]
    local_filepath = pathlib.Path(checkpoints_directory, local_filename)
    local_filepath.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading {url} ...")
    with httpx.stream("GET", url, follow_redirects=True) as stream:
        total = int(stream.headers["Content-Length"])
        with open(local_filepath, "wb") as f, tqdm(
            total=total, unit_scale=True, unit_divisor=1024, unit="B"
        ) as progress:
            num_bytes_downloaded = stream.num_bytes_downloaded
            for data in stream.iter_bytes():
                f.write(data)
                progress.update(
                    stream.num_bytes_downloaded - num_bytes_downloaded
                )
                num_bytes_downloaded = stream.num_bytes_downloaded


# Pin to a specific commit from https://github.com/comfyanonymous/ComfyUI/commits/master/
# for stability. To update to a later ComfyUI version, change this commit identifier.
comfyui_commit_sha = "b3b5ddb07a23b3d070df292c7a7fd6f83dc8fd50"

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    # Here we place the latest ComfyUI repository code into /root.
    # Because /root is almost empty, but not entirely empty
    # as it contains this comfy_ui.py script, `git clone` won't work.
    # As a workaround we `init` inside the non-empty directory, then `checkout`.
    .run_commands(
        "cd /root && git init .",
        "cd /root && git remote add --fetch origin https://github.com/comfyanonymous/ComfyUI",
        f"cd /root && git checkout {comfyui_commit_sha}",
        "cd /root && pip install xformers!=0.0.18 -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121",
    )
    # Use fork of https://github.com/valohai/asgiproxy with bugfixes.
    .pip_install(
        "git+https://github.com/modal-labs/asgiproxy.git@ef25fe52cf226f9a635e87616e7c049e451e2bd8",
        "httpx",
        "tqdm",
    )
    .run_function(download_checkpoint)
)
stub = modal.Stub(name="example-comfy-ui", image=image)

# ## Start the ComfyUI server
#
# Inside the container, we will run the ComfyUI server and execution queue on port 8188. Then, we
# wrap this function in the `@web_server` decorator to expose the server as a web endpoint.
#
# For ASGI-compatible frameworks, you can also use Modal's `@asgi_app` decorator.


@stub.function(
    gpu="any",
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    # Restrict to 1 container because we want to our ComfyUI session state
    # to be on a single container.
    concurrency_limit=1,
    timeout=1800,
)
@modal.web_server(8188, startup_timeout=30)
def web():
    cmd = "python main.py --dont-print-server --multi-user --listen --port 8188"
    subprocess.Popen(cmd, shell=True)
